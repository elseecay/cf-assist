from undetected_chromedriver import Chrome, ChromeOptions
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

import bs4
import lxml

import argparse
import pickle
import os
import socket
import re
import logging
import tempfile

from contextlib import closing
from pathlib import Path
from time import sleep
from typing import Any
from collections import namedtuple
from typing import Tuple


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
PORT = 58610
LOG_FILE = "cf_browser.log"


driver: Chrome = None
sock: socket.socket = None
logger: logging.Logger = None


class CannotSolveCaptcha(Exception):
    pass


def init_socket():
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', PORT))
    sock.listen(1)
    logger.info(f"Listen on {PORT}")


def is_port_open():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("localhost", PORT))
    except ConnectionError as e:
        return True
    sock.close()
    return False


def init_driver(window: bool):
    global driver
    options = ChromeOptions()
    options.add_argument(f"--user-agent={USER_AGENT}")
    driver = Chrome(headless=not window, use_subprocess=False, options=options)
    driver.set_window_size(1920, 1080)
    logger.info(f"Chrome started, window={window}")


def init_logging():
    global logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(formatter)
    logger = logging.Logger("cf_browser", level=logging.DEBUG)
    logger.addHandler(handler)


def read_request(client):
    length = int.from_bytes(client.recv(4), byteorder="little")
    data = client.recv(length)
    request = pickle.loads(data)
    logger.info("Request received: %s", request["cmd"])
    return request


def write_response(client, response):
    data = pickle.dumps(response)
    client.send(len(data).to_bytes(length=4, byteorder="little"))
    client.send(data)
    logger.info("Response sent: %s", response)


def handle_request(request: dict) -> Any:
    handler = globals()["handle_" + request["cmd"]]
    try:
        result = handler(request)
    except Exception as e:
        logger.warning("(%s) Request handling failed, %s", request["cmd"], e)
        return Exception(str(e))
    return result


def wait_element(by: Any, value: Any):
    sleep(0.25)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((by, value)))
    sleep(0.25)


def wait_complete():
    sleep(0.25)
    is_complete = lambda _: driver.execute_script("return document.readyState;") == "complete"
    WebDriverWait(driver, 30).until(is_complete)
    sleep(0.25)


def is_captcha():
    try:
        elem = driver.find_element(By.CLASS_NAME, "main-wrapper")
        elem = elem.find_element(By.CLASS_NAME, "main-content")
        return True
    except NoSuchElementException:
        return False
    

def handle_stop_browser(request: dict) -> int:
    return 0
    

def handle_participants_count(request: dict) -> int:
    contest_id = request["contest_id"]
    open_page(f"https://codeforces.com/contest/{contest_id}/standings")
    wait_element(By.CLASS_NAME, "page-index")
    elem = driver.find_element(By.CLASS_NAME, "custom-links-pagination")
    children = elem.find_elements(By.XPATH, "./*")
    elem = children[-1].find_element(By.TAG_NAME, "a")
    matches = re.findall(r"\d+", elem.text)
    return int(matches[-1])


def handle_submit_problem(request: dict) -> int:
    contest_id = request["contest_id"]
    problem = request["problem"]
    cf_compiler = request["cf_compiler"]
    code = request["code"]
    file_path = tempfile.mktemp()
    with open(file_path, "w") as f:
        f.write(code)
    open_page(f"https://codeforces.com/contest/{contest_id}/submit")
    wait_complete()
    wait_element(By.ID, "singlePageSubmitButton")
    tag = driver.find_element(By.NAME, "submittedProblemIndex")
    select = Select(tag)
    select.select_by_value(problem)
    tag = driver.find_element(By.NAME, "programTypeId")
    select = Select(tag)
    select.select_by_value(cf_compiler)
    tag = driver.find_element(By.NAME, "sourceFile")
    tag.send_keys(file_path)
    button = driver.find_element(By.ID, "singlePageSubmitButton")
    button_enabled = lambda _: driver.execute_script("return document.getElementById(\"singlePageSubmitButton\").disabled == false;")
    WebDriverWait(driver, 30).until(button_enabled)
    button.click()
    wait_complete()
    if "You have submitted exactly the same code before" in driver.page_source:
        raise Exception("You have submitted exactly the same code before")
    wait_element(By.CLASS_NAME, "datatable")
    return 0


def handle_test_sample(request: dict) -> Tuple[str, str]:
    contest_id = request["contest_id"]
    problem = request["problem"]
    index = request["index"]
    open_page(f"https://codeforces.com/contest/{contest_id}/problem/{problem}")
    wait_element(By.CLASS_NAME, "sample-test")
    soup = bs4.BeautifulSoup(driver.page_source, "lxml")
    sample_node = soup.find("div", {"class": "sample-test"})
    if not sample_node:
        raise Exception("Sample block not found")
    if index < 0 or index * 2 + 1 >= len(sample_node.contents):
        raise Exception("No such sample index")
    input_block = sample_node.contents[2 * index].find("pre")
    output_block = sample_node.contents[2 * index + 1].find("pre")
    input_text = "\n".join(e.text.strip() for e in input_block.contents if len(e.text) > 0)
    output_text = "\n".join(e.text.strip() for e in output_block.contents if len(e.text) > 0)
    return input_text, output_text


def enter(login: str, password: str) -> bool:
    open_page("https://codeforces.com/enter")
    wait_element(By.ID, "handleOrEmail")
    login_elem = driver.find_element(By.ID, "handleOrEmail")
    login_elem.send_keys(login)
    password_elem = driver.find_element(By.ID, "password")
    password_elem.send_keys(password)
    remember = driver.find_element(By.ID, "remember")
    remember.click()
    button = driver.find_element(By.CLASS_NAME, "submit")
    sleep(1)
    button.click()
    sleep(1)
    wait_complete()
    open_page(f"https://codeforces.com/profile/{login}")
    try:
        driver.find_element(By.XPATH, "//*[@href='/register']")
    except NoSuchElementException:
        return True
    return False


def open_page(url: str):
    logger.info("Opening page %s", url)
    driver.get(url)
    captcha = is_captcha()
    if captcha:
        logger.info("There is a captcha")
        solve_captcha()
        wait_complete()
        driver.get(url)
    if captcha:
        if is_captcha():
            logger.critical("Cannot solve captcha")
            raise CannotSolveCaptcha(f"URL: {url}")
        logger.info("Captcha solved!")
    wait_complete()


def solve_captcha():
    wait_element(By.CLASS_NAME, "main-content")
    sleep(5)
    main_content = driver.find_element(By.CLASS_NAME, "main-content")
    captcha_div = main_content.find_element(By.XPATH, f"./* [3]")
    x_size, y_size = captcha_div.size["width"], captcha_div.size["height"]
    actions = ActionChains(driver, duration=2000)
    actions.move_to_element(captcha_div)
    actions.move_by_offset(-x_size // 2, -y_size // 2)
    actions.move_by_offset(40, 40)
    actions.click()
    actions.perform()
    sleep(3)
    wait_complete()


def start_browser(login, password, window):
    init_logging()
    if not is_port_open():
        logger.critical("Port %s is already open", PORT)
        return
    init_socket()
    init_driver(window)
    for _ in range(5):
        logged_in = False
        try:
            if enter(login, password):
                logged_in = True
                break
            else:
                logger.warning("Enter failed, invalid login/password?")
        except Exception as e:
            logger.critical("Enter failed, %s", e)
        logger.warning("Sleeping 10 seconds")
        sleep(10)
    if not logged_in:
        logger.critical("Cannot enter after 5 attempts, exiting")
        return
    logger.info("Enter successful!")
    while True:
        driver.get("about:blank")
        client, _ = sock.accept()
        try:
            request = read_request(client)
            response = handle_request(request)
            write_response(client, response)
            if request["cmd"] == "stop_browser":
                break
        except Exception as e:
            logger.critical("Request failed, %s", e)
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser("cf_browser.py", formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35))
    argparser.add_argument("-l", "--login", type=str, required=True, help="Codeforces login")
    argparser.add_argument("-p", "--password", type=str, required=True, help="Codeforces password")
    argparser.add_argument("-w", "--window", action="store_true", help="Show browser window")
    return argparser.parse_args()


def main():
    args = parse_args()
    try:
        start_browser(args.login, args.password, args.window)
    except Exception as e:
        if logger:
            logger.critical("Terminating service, %s", e)
    finally:
        if driver:
            driver.quit()
        if sock:
            sock.close()


if __name__ == "__main__":
    main()
