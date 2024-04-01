# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script leverages the GPT-4V API and Playwright to create a web agent capable of autonomously performing tasks on webpages.
It utilizes Playwright to create browser and retrieve interactive elements, then apply [SeeAct Framework](https://osu-nlp-group.github.io/SeeAct/) to generate and ground the next operation.
The script is designed to automate complex web interactions, enhancing accessibility and efficiency in web navigation tasks.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import warnings
from dataclasses import dataclass

import toml
import torch
from aioconsole import ainput, aprint
from playwright.async_api import async_playwright

from data_utils.format_prompt_utils import get_index_from_option_name
from data_utils.prompts import generate_prompt
from data_utils.format_prompt_utils import format_options
from demo_utils.browser_helper import (normal_launch_async, normal_new_context_async,
                                       get_interactive_elements_with_playwright, select_option, saveconfig)
from demo_utils.format_prompt import format_choices, format_ranking_input, postprocess_action_lmm
from demo_utils.inference_engine import OpenaiEngine
from demo_utils.ranking_model import CrossEncoder, find_topk
from demo_utils.website_dict import website_dict

# Remove Huggingface internal warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


#cdp- Chrome Devtools Protocol
@dataclass
class SessionControl:
    pages = []
    cdp_sessions = []
    active_page = None
    active_cdp_session = None
    context = None
    browser = None


session_control = SessionControl()


#
# async def init_cdp_session(page):
#     cdp_session = await page.context.new_cdp_session(page)
#     await cdp_session.send("DOM.enable")
#     await cdp_session.send("Overlay.enable")
#     await cdp_session.send("Accessibility.enable")
#     await cdp_session.send("Page.enable")
#     await cdp_session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
#     return cdp_session


async def page_on_close_handler(page):
    # print("Closed: ", page)
    if session_control.context:
        # if True:
        try:
            #what was the point of this?
            await session_control.active_page.title()
            # print("Current active page: ", session_control.active_page)
        except:
            #todo need to figure out how to port some of this logic to the chrome extension (or which parts to ignore)
            # I think it's irrelevant, b/c the agent loop will be running in code in an injected content script,
            # which would be running in the context of the page itself and so would die if the tab was closed
            await aprint("The active tab was closed. Will switch to the last page (or open a new default google page)")
            # print("All pages:")
            # print('-' * 10)
            # print(session_control.context.pages)
            # print('-' * 10)
            if session_control.context.pages:
                session_control.active_page = session_control.context.pages[-1]
                await session_control.active_page.bring_to_front()
                await aprint("Switched the active tab to: ", session_control.active_page.url)
            else:
                await session_control.context.new_page()
                try:
                    await session_control.active_page.goto("https://www.google.com/", wait_until="load")
                except Exception as e:
                    pass
                await aprint("Switched the active tab to: ", session_control.active_page.url)


async def page_on_navigatio_handler(frame):
    session_control.active_page = frame.page
    # print("Page navigated to:", frame.url)
    # print("The active tab is set to: ", frame.page.url)


# https://playwright.dev/docs/api/class-page#page-event-crash
async def page_on_crash_handler(page):
    await aprint("Page crashed:", page.url)
    await aprint("Try to reload")
    page.reload()
    #todo? figure out how to port this? can't see how the js in the page could recover from the page as a whole crashing


async def page_on_open_handler(page):
    # print("Opened: ",page)
    page.on("framenavigated", page_on_navigatio_handler)
    page.on("close", page_on_close_handler)
    page.on("crash", page_on_crash_handler)
    session_control.active_page = page
    # print("The active tab is set to: ", page.url)
    # print("All pages:")
    # print('-'*10)
    # print(session_control.context.pages)
    # print("active page: ",session_control.active_page)
    # print('-' * 10)


async def main(config, base_dir) -> None:
    fixed_choice_batch_size = config["experiment"]["fixed_choice_batch_size"]
    dynamic_choice_batch_size = config["experiment"]["dynamic_choice_batch_size"]
    max_continuous_no_op = config["experiment"]["max_continuous_no_op"]
    max_op = config["experiment"]["max_op"]
    highlight = config["experiment"]["highlight"]
    monitor = config["experiment"]["monitor"]
    dev_mode = config["experiment"]["dev_mode"]
    # openai settings
    openai_config = config["openai"]

    # Initialize Inference Engine based on OpenAI API
    generation_model = OpenaiEngine(**openai_config, )

    task_input = await ainput(f"Please input a task, and press Enter.\nTask: ")
    #I assume that, in the browser extension, the starting website will simply be whatever is open in the current tab
    # set the folder name as current time
    #todo browser code should also incorporate a task or run id based on timestamp at very start of run
    # and put that in logs (can modify the augmentLogMsg() function)
    # determine at start of content script (then put as global variable in window object)
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    confirmed_task = task_input
    task_id = file_name

    # init logger
    logger = logging.getLogger(f"{task_id}")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    if dev_mode:
        logger.setLevel(logging.DEBUG)

    #todo debug logging about config choices (e.g. max op, monitor/dev flags)

    logger.info(f"website: todo get website from current tab")
    logger.info(f"task: {confirmed_task}")
    logger.info(f"id: {task_id}")
    #removing stuff related to initial page opening/loading b/c extension will be triggered when the target starting
    # page is already loaded as the current tab
    taken_actions = []
    complete_flag = False
    monitor_signal = ""
    time_step = 0
    no_op_count = 0
    valid_op_count = 0

    while not complete_flag:
        logger.debug(f"Page at the start: todo get url from curr tab")
        log_separator_line(logger)
        logger.info(f"Time step: {time_step}")
        log_separator_line(logger)
        elements = await get_interactive_elements_with_playwright(session_control.active_page)

        logger.info(f"# all interactive elements: {len(elements)}")
        for i in elements:
            logger.debug(i[1:])
        time_step += 1

        if len(elements) == 0:
            if monitor:
                logger.info(
                    f"----------There is no element in this page. Do you want to terminate or continue after"
                    f"human intervention? [i/e].\ni(Intervene): Reject this action, and pause for human "
                    f"intervention.\ne(Exit): Terminate the program and save results.")
                monitor_input = await ainput()
                logger.info("Monitor Command: " + monitor_input)
                if monitor_input in ["i", "intervene", 'intervention']:
                    logger.info(
                        "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                    human_intervention = await ainput()
                    if human_intervention:
                        human_intervention = f"Human intervention with a message: {human_intervention}"
                    else:
                        human_intervention = f"Human intervention"
                    taken_actions.append(human_intervention)
                    continue

            logger.info("Terminate because there is no interactive element in this page.")
            #todo evaluate if this and code block from around line 700 can be consolidated into one section after the
            # end of the while loop
            logger.info("Action History:")#maybe simplify this and next 2 lines with "\n" + taken_actions.join("\n") concatenated to heading
            for action in taken_actions:
                logger.info(action)
            # can simplify next 3 lines with ternary in final json spec
            success_or_not = ""
            if valid_op_count == 0:
                success_or_not = "0"
            final_json = {"confirmed_task": confirmed_task, "website": "todo get url from curr tab",
                          "task_id": task_id, "success_or_not": success_or_not,
                          "num_step": len(taken_actions), "action_history": taken_actions,
                          "exit_by": "No elements"}
            logger.info("Final JSON:" + json.dumps(final_json, indent=4))
            continue

        all_candidate_ids = range(len(elements))
        ranked_elements = elements
        all_candidate_ids_with_location = []
        for element_id, element_detail in zip(all_candidate_ids, ranked_elements):
            all_candidate_ids_with_location.append(
                (element_id, round(element_detail[0][1]), round(element_detail[0][0])))

        all_candidate_ids_with_location.sort(key=lambda x: (x[1], x[2]))
        all_candidate_ids = [element_id[0] for element_id in all_candidate_ids_with_location]
        num_choices = len(all_candidate_ids)
        total_height = await session_control.active_page.evaluate('''() => {
                                                        return Math.max(
                                                            document.documentElement.scrollHeight, 
                                                            document.body.scrollHeight,
                                                            document.documentElement.clientHeight
                                                        );
                                                    }''')
        if dynamic_choice_batch_size > 0:
            step_length = min(num_choices,
                              num_choices // max(round(total_height / dynamic_choice_batch_size), 1) + 1)
        else:
            step_length = min(num_choices, fixed_choice_batch_size)
        logger.info(f"batch size: {step_length}")
        log_separator_line(logger)

        total_width = session_control.active_page.viewport_size["width"]
        logger.info("You are asked to complete the following task: " + confirmed_task)
        previous_actions = taken_actions

        previous_action_text = "Previous Actions:\n"
        if previous_actions is None or previous_actions == []:## can simplify this if block with nullish coalescing
            previous_actions = ["None"]
        for action_text in previous_actions:#can simplify next 4 lines using 1 concat and string join or equivalent
            previous_action_text += action_text
            previous_action_text += "\n"
        logger.info(previous_action_text[:-1])

        target_element = []#various bits of info about the target element
        new_action = ""
        # todo is there a reason to have a default action like this? don't we treat the round as a no-op
        #  if the llm didn't spit out a valid action name?
        target_action = "CLICK"
        target_value = ""
        query_count = 0
        #todo isn't this boolean flag redundant? either branch which sets it will also put a non-empty value in
        # target_action (this reasoning would only work if target_action stops having a default value of CLICK)
        got_one_answer = False

        #todo at least 1 separate method for element selection, with it also possibly split into sub-methods
        # block is ~100 lines long
        for multichoice_i in range(0, num_choices, step_length):
            logger.info("-" * 10)
            logger.info(f"Start Multi-Choice QA - Batch {multichoice_i // step_length}")

            height_start = all_candidate_ids_with_location[multichoice_i][1]
            height_end = all_candidate_ids_with_location[min(multichoice_i + step_length, num_choices) - 1][1]
            total_height = await session_control.active_page.evaluate('''() => {
                                                            return Math.max(
                                                                document.documentElement.scrollHeight, 
                                                                document.body.scrollHeight,
                                                                document.documentElement.clientHeight);
                                                        }''')
            clip_start = min(total_height - 1144, max(0, height_start - 200))
            clip_height = min(total_height - clip_start, max(height_end - height_start + 200, 1144))
            clip = {"x": 0, "y": clip_start, "width": total_width, "height": clip_height}

            logger.debug(height_start)
            logger.debug(height_end)
            logger.debug(total_height)
            logger.debug(clip)
            logger.debug(multichoice_i)
            candidate_ids = all_candidate_ids[multichoice_i:multichoice_i + step_length]
            choices = format_choices(elements, candidate_ids, confirmed_task, taken_actions)
            query_count += 1
            # Format prompts for LLM inference
            prompt = generate_prompt(task=confirmed_task, previous=taken_actions, choices=choices,
                                     experiment_split="SeeAct")
            for prompt_i in prompt:
                logger.debug(prompt_i)

            output0 = generation_model.generate(prompt=prompt, image_path="will be data url, not file path", turn_number=0)
            log_separator_line(logger)
            logger.debug("🤖Action Generation Output🤖")
            # why can't the entire output (containing newlines) simply be passed to the logger at once?
            for line in output0.split('\n'):
                logger.info(line)
            log_separator_line(logger)

            choice_text = f"(Multichoice Question) - Batch {multichoice_i // step_length}" + "\n" + format_options(choices)
            choice_text = choice_text.replace("\n\n", "")#why?
            for line in choice_text.split('\n'):
                logger.info(line)
            # why can't the entire output (containing newlines) simply be passed to the logger at once?

            output = generation_model.generate(prompt=prompt, image_path="will be data url, not file path", turn_number=1,
                                               ouput__0=output0)
            log_separator_line(logger)
            logger.info("🤖Grounding Output🤖")
            # why can't the entire output (containing newlines) simply be passed to the logger at once?
            for line in output.split('\n'):
                logger.info(line)
            pred_element, pred_action, pred_value = postprocess_action_lmm(output)
            #todo can simplify this code by modifying get_index_from_option_name so that,  if input is bad,
            # it returns -1 rather than throwing
            if len(pred_element) in [1, 2]:
                element_id = get_index_from_option_name(pred_element)
            else:
                element_id = -1

            # Process the elements
            if (0 <= element_id < len(candidate_ids) and pred_action.strip() in ["CLICK", "SELECT", "TYPE",
                                                                                 "PRESS ENTER", "HOVER",
                                                                                 "TERMINATE"]):
                target_element = elements[int(choices[element_id][0])]
                target_element_text = choices[element_id][1]
                target_action = pred_action
                target_value = pred_value
                new_action += "[" + target_element[2] + "]" + " "
                new_action += target_element[1] + " -> " + target_action
                if target_action.strip() in ["SELECT", "TYPE"]:
                    new_action += ": " + target_value
                got_one_answer = True
                break
            elif pred_action.strip() in ["PRESS ENTER", "TERMINATE"]:
                target_element = pred_action
                target_element_text = target_element
                target_action = pred_action
                target_value = pred_value
                new_action += target_action
                if target_action.strip() in ["SELECT", "TYPE"]:
                    new_action += ": " + target_value
                got_one_answer = True
                break
            else:
                pass

        if got_one_answer:
            log_separator_line(logger)
            logger.info("🤖Browser Operation🤖")
            logger.info(f"Target Element: {target_element_text}", )
            logger.info(f"Target Action: {target_action}", )
            logger.info(f"Target Value: {target_value}", )

            if monitor:
                logger.info(
                    f"----------\nShould I execute the above action? [Y/n/i/e].\nY/n: Accept or reject this action.\ni(Intervene): Reject this action, and pause for human intervention.\ne(Exit): Terminate the program and save results.")
                monitor_input = await ainput()
                logger.info("Monitor Command: " + monitor_input)
                if monitor_input in ["n", "N", "No", "no"]:
                    monitor_signal = "reject"
                    target_element = []
                elif monitor_input in ["e", "exit", "Exit"]:
                    monitor_signal = "exit"
                elif monitor_input in ["i", "intervene", 'intervention']:
                    monitor_signal = "pause"
                    target_element = []
                else:
                    valid_op_count += 1
        else:
            no_op_count += 1
            target_element = []

        try:
            if monitor_signal == 'exit':
                raise Exception("human supervisor manually made it exit.")
            if no_op_count >= max_continuous_no_op:
                raise Exception(f"no executable operations for {max_continuous_no_op} times.")
            elif time_step >= max_op:
                raise Exception(f"the agent reached the step limit {max_op}.")
            elif target_action == "TERMINATE":
                raise Exception("The model determined a completion.")

            # Perform browser action with PlayWright
            # The code is complex to handle all kinds of cases in execution
            # It's ugly, but it works, so far
            selector = None
            fail_to_execute = False
            try:
                if target_element == []:
                    pass
                else:
                    if not target_element in ["PRESS ENTER", "TERMINATE"]:
                        selector = target_element[-2]
                        logger.debug(target_element)
                        try:
                            #todo how to scroll to element in js?
                            await selector.scroll_into_view_if_needed(timeout=3000)
                            if highlight:
                                #todo how to highlight element with js?
                                # https://playwright.dev/docs/api/class-locator#locator-highlight
                                # docs say not to commit code that uses this function?
                                # what does it even do (as distinct from hover)?
                                await selector.highlight()
                                await asyncio.sleep(2.5)
                        except Exception as e:
                            pass

                #todo consider implementing in js a conditional polling/wait for 'stability'
                # https://playwright.dev/docs/actionability#stable
                #todo might also need to implement a 'receives events' polling check
                # https://playwright.dev/docs/actionability#receives-events
                #todo note that a given action type would only need some of the actionability checks
                # https://playwright.dev/docs/actionability#introduction
                #todo above goals for conditional polling/waiting could be one (or a few) helper methods

                # todo after chat with Boyuan to clarify questions in below big block, figure out how to break it up
                if selector:
                    valid_op_count += 1
                    if target_action == "CLICK":
                        js_click = True
                        try:
                            if target_element[-1] in ["select", "input"]:
                                logger.info("Try performing a CLICK")
                                await selector.evaluate("element => element.click()", timeout=10000)
                                js_click = False
                            else:
                                #todo this automatically does a scroll-to-element
                                # also need special tricks to approximately simulate click-at-coordinates with js
                                # https://stackoverflow.com/questions/3277369/how-to-simulate-a-click-by-using-x-y-coordinates-in-javascript
                                # https://javascript.plainenglish.io/how-to-simulate-a-click-by-using-x-y-coordinates-in-javascript-82b745e4a6b1
                                await selector.click(timeout=10000)
                        except Exception as e:
                            try:
                                if not js_click:
                                    logger.info("Try performing a CLICK")
                                    await selector.evaluate("element => element.click()", timeout=10000)
                                else:
                                    raise Exception(e)
                            except Exception as ee:
                                try:
                                    logger.info("Try performing a HOVER")
                                    await selector.hover(timeout=10000)
                                    # for hover/mouseover: https://stackoverflow.com/a/50206746/10808625
                                    new_action = new_action.replace("CLICK",
                                                                    f"Failed to CLICK because {e}, did a HOVER instead")
                                except Exception as eee:
                                    new_action = new_action.replace("CLICK", f"Failed to CLICK because {e}")
                                    no_op_count += 1
                    elif target_action == "TYPE":
                        try:
                            try:
                                logger.info("Try performing a \"press_sequentially\"")
                                await selector.clear(timeout=10000)# js can set element.value to empty string
                                await selector.fill("", timeout=10000)
                                #why are we filling with empty string after we already cleared it?
                                await selector.press_sequentially(target_value, timeout=10000)
                                #todo ask Boyuan why we're using press_sequentially instead of fill
                            except Exception as e0:
                                await selector.fill(target_value, timeout=10000)
                        except Exception as e:
                            try:
                                #can we please check whether it's a select before trying to type into it?
                                # or does this typing work sometimes?
                                if target_element[-1] in ["select"]:
                                    logger.info("Try performing a SELECT")
                                    selected_value = await select_option(selector, target_value)
                                    new_action = new_action.replace("TYPE",
                                                                    f"Failed to TYPE \"{target_value}\" because {e}, did a SELECT {selected_value} instead")
                                else:
                                    raise Exception(e)
                            except Exception as ee:
                                #does this variable mean "we haven't completed an attempt at a js click yet"?
                                # if so, why do we only try clicking again in an except block if we had already clicked?
                                js_click = True
                                try:
                                    if target_element[-1] in ["select", "input"]:
                                        logger.info("Try performing a CLICK")
                                        await selector.evaluate("element => element.click()", timeout=10000)
                                        js_click = False
                                    else:
                                        logger.info("Try performing a CLICK")
                                        await selector.click(timeout=10000)
                                    new_action = "[" + target_element[2] + "]" + " "
                                    new_action += target_element[
                                                      1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                except Exception as eee:
                                    try:
                                        if not js_click:
                                            logger.debug(eee)
                                            logger.info("Try performing a CLICK")
                                            await selector.evaluate("element => element.click()", timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a CLICK instead"
                                        else:
                                            raise Exception(eee)
                                    except Exception as eeee:
                                        try:
                                            #very confused, why are we doing a hover as the absolute last resort only for the TYPE action?
                                            logger.info("Try performing a HOVER")
                                            await selector.hover(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}, did a HOVER instead"
                                        except Exception as eee:
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to TYPE \"{target_value}\" because {e}"
                                            no_op_count += 1
                    elif target_action == "SELECT":
                        try:
                            logger.info("Try performing a SELECT")
                            selected_value = await select_option(selector, target_value)
                            new_action = new_action.replace(f"{target_value}", f"{selected_value}")
                        except Exception as e:
                            try:
                                #there are cases where the action type is SELECT but the element is an input??
                                if target_element[-1] in ["input"]:
                                    try:
                                        logger.info("Try performing a \"press_sequentially\"")
                                        await selector.clear(timeout=10000)
                                        await selector.fill("", timeout=10000)
                                        await selector.press_sequentially(target_value, timeout=10000)
                                    except Exception as e0:
                                        await selector.fill(target_value, timeout=10000)
                                    new_action = new_action.replace("SELECT",
                                                                    f"Failed to SELECT \"{target_value}\" because {e}, did a TYPE instead")
                                else:
                                    raise Exception(e)
                            except Exception as ee:
                                js_click = True
                                try:
                                    if target_element[-1] in ["select", "input"]:
                                        logger.info("Try performing a CLICK")
                                        await selector.evaluate("element => element.click()", timeout=10000)
                                        js_click = False
                                        #why are we doing js click for select and input elements but the playwright
                                        # click for others (in the scenario where the action name is SELECT)?
                                    else:

                                        await selector.click(timeout=10000)
                                    new_action = "[" + target_element[2] + "]" + " "
                                    new_action += target_element[
                                                      1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a CLICK instead"
                                except Exception as eee:
                                    try:
                                        if not js_click:
                                            logger.info("Try performing a CLICK")
                                            await selector.evaluate("element => element.click()", timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a CLICK instead"
                                        else:
                                            raise Exception(eee)
                                    except Exception as eeee:
                                        try:
                                            logger.info("Try performing a HOVER")
                                            await selector.hover(timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}, did a HOVER instead"
                                        except Exception as eee:
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to SELECT \"{target_value}\" because {e}"
                                            no_op_count += 1
                    elif target_action == "HOVER":
                        try:
                            logger.info("Try performing a HOVER")
                            await selector.hover(timeout=10000)
                        except Exception as e:
                            try:
                                await selector.click(timeout=10000)
                                new_action = new_action.replace("HOVER",
                                                                f"Failed to HOVER because {e}, did a CLICK instead")
                            except:
                                js_click = True
                                try:
                                    if target_element[-1] in ["select", "input"]:
                                        logger.info("Try performing a CLICK")
                                        await selector.evaluate("element => element.click()", timeout=10000)
                                        js_click = False
                                    else:
                                        await selector.click(timeout=10000)
                                    new_action = "[" + target_element[2] + "]" + " "
                                    new_action += target_element[
                                                      1] + " -> " + f"Failed to HOVER because {e}, did a CLICK instead"
                                except Exception as eee:
                                    try:
                                        if not js_click:
                                            logger.info("Try performing a CLICK")
                                            await selector.evaluate("element => element.click()", timeout=10000)
                                            new_action = "[" + target_element[2] + "]" + " "
                                            new_action += target_element[
                                                              1] + " -> " + f"Failed to HOVER because {e}, did a CLICK instead"
                                        else:
                                            raise Exception(eee)
                                    except Exception as eeee:
                                        new_action = "[" + target_element[2] + "]" + " "
                                        new_action += target_element[
                                                          1] + " -> " + f"Failed to HOVER because {e}"
                                        no_op_count += 1
                    elif target_action == "PRESS ENTER":
                        try:
                            logger.info("Try performing a PRESS ENTER")
                            await selector.press('Enter')
                            #todo need to e2e test the js code that replaces this
                            # https://www.shecodes.io/athena/72484-how-to-simulate-pressing-the-enter-key-in-javascript
                            # https://playwright.dev/docs/api/class-locator#locator-press
                        except Exception as e:
                            await selector.click(timeout=10000)
                            await session_control.active_page.keyboard.press('Enter')
                            #https://playwright.dev/docs/api/class-keyboard#keyboard-press
                            # how does this differ from locator.press()? not sure how to replicate this as separate
                            # thing in js
                elif monitor_signal == "pause":
                    logger.info(
                        "Pause for human intervention. Press Enter to continue. You can also enter your message here, which will be included in the action history as a human message.")
                    human_intervention = await ainput()
                    if human_intervention:
                        human_intervention = f" Human message: {human_intervention}"
                    raise Exception(
                        f"the human supervisor rejected this operation and may have taken some actions.{human_intervention}")
                elif monitor_signal == "reject":
                    raise Exception("the human supervisor rejected this operation.")
                elif target_element == "PRESS ENTER":#wth? the element is PRESS ENTER???
                    logger.info("Try performing a PRESS ENTER")
                    await session_control.active_page.keyboard.press('Enter')
                no_op_count = 0#why are we resetting the no op count here?
                try:
                    await session_control.active_page.wait_for_load_state('load')
                    # https://stackoverflow.com/questions/1033398/how-to-execute-a-function-when-page-has-fully-loaded
                except Exception as e:
                    pass
            except Exception as e:
                if target_action not in ["TYPE", "SELECT"]:
                    new_action = f"Failed to {target_action} {target_element_text} because {e}"
                else:
                    new_action = f"Failed to {target_action} {target_value} for {target_element_text} because {e}"
                fail_to_execute = True

            if new_action == "" or fail_to_execute:
                if new_action == "":
                    new_action = "No Operation"
                if monitor_signal not in ["pause", "reject"]:
                    no_op_count += 1
            taken_actions.append(new_action)
            if not session_control.context.pages:
                await session_control.context.new_page()
                try:
                    #todo ask if should just cut this, not sure how it would be relevant in non-playwright context
                    await session_control.active_page.goto("starting url??", wait_until="load")
                except Exception as e:
                    pass

            # todo need to figure out js equivalent for waiting until page load after action?
            # https://stackoverflow.com/questions/1033398/how-to-execute-a-function-when-page-has-fully-loaded

            if monitor_signal == 'pause':
                pass# todo why do we skip the 3sec sleep if we're pausing?
            else:
                await asyncio.sleep(3)
            logger.debug(f"current active page: {session_control.active_page}")

            logger.debug("All pages")
            logger.debug(session_control.context.pages)
            logger.debug("-" * 10)
            try:
                await session_control.active_page.wait_for_load_state('load')
            except Exception as e:
                logger.debug(e)
        except Exception as e:
            logger.info("=" * 10)
            logger.info(f"Decide to terminate because {e}")
            logger.info("Action History:")
            for action in taken_actions:
                logger.info(action)

            success_or_not = ""
            if valid_op_count == 0:
                success_or_not = "0"
            final_json = {"confirmed_task": confirmed_task, "website": "todo starting url",
                          "task_id": task_id, "success_or_not": success_or_not,
                          "num_step": len(taken_actions), "action_history": taken_actions, "exit_by": str(e)}
            logger.info("Final JSON:" + json.dumps(final_json, indent=4))
            if monitor:
                logger.info("Wait for human inspection. Directly press Enter to exit")
                monitor_input = await ainput()

            logger.info("Close browser context")
            logger.removeHandler(console_handler)
            complete_flag = True


def log_separator_line(logger):
    terminal_width = 10
    logger.info("-" * terminal_width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", help="Path to the TOML configuration file.", type=str, metavar='config',
                        default=f"{os.path.join('config', 'demo_mode.toml')}")
    args = parser.parse_args()

    # Load configuration file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = None
    try:
        with open(os.path.join(base_dir, args.config_path) if not os.path.isabs(args.config_path) else args.config_path,
                  'r') as toml_config_file:
            config = toml.load(toml_config_file)
            print(f"Configuration File Loaded - {os.path.join(base_dir, args.config_path)}")
    except FileNotFoundError:
        print(f"Error: File '{args.config_path}' not found.")
        #todo I'm confused- why doesn't this have a raise statement after the print? same with next except block
    except toml.TomlDecodeError:
        print(f"Error: File '{args.config_path}' is not a valid TOML file.")

    asyncio.run(main(config, base_dir))
