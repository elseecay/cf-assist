import subprocess
import os
import sys
import importlib
import inspect
import platform
import functools
import sqlite3
import random
import hashlib
import json
import time
import string
import re
import socket
import pickle
import argparse

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Tuple, Union
from enum import StrEnum

import colorama
import requests


ROOT = Path(__file__).parent.absolute()
SOURCE_FILE = Path(ROOT, "main.cpp")
EXECUTABLE_FILE = Path(ROOT, "main.exe")
INPUT_FILE = Path(ROOT, "input.txt")
OUTPUT_FILE = Path(ROOT, "output.txt")
EXPECTED_FILE = Path(ROOT, "expected.txt")
SETTINGS_FILE = Path(ROOT, "cf.settings")
SOURCE_FILE_BACKUP = Path(ROOT, "main.cpp.backup")


TIMEOUT_DEFAULT = (20.0, 60.0)
BROWSER_PORT = 58610


class Error(Exception):
    pass


class RequestError(Error):

    def __init__(self, message: str, obj: Union[requests.PreparedRequest, requests.Response]):
        super().__init__(message)
        if isinstance(obj, requests.Response):
            self.response = obj
            self.request = obj.request
        elif isinstance(obj, requests.PreparedRequest):
            self.request = obj
            self.response = None
        else:
            assert False


class CompilationError(Error):

    def __init__(self, compilation_string: str):
        super().__init__("compilation error")
        self.compilation_string = compilation_string


class ProblemError(Error):
    pass


class ExecutionError(Error):
    pass


@dataclass
class SampleTest:
    input: str
    output: str


@dataclass
class Contest:
    name: str
    id: int
    start_time: datetime


@dataclass
class Submission:
    contest_id: int
    problem: str
    tests_passed: int
    creation_timestamp: int
    verdict: Optional[str] = None
    time_ms: Optional[int] = None
    mem_mb: Optional[int] = None


@dataclass
class User:
    handle: str
    rating: int
    max_rating: int


@dataclass
class Contestant:
    handle: str
    place: int


class CodeforcesCompiler(StrEnum):
    GPP17 = "54"
    GPP20 = "89"
    GPP23 = "91"


@dataclass
class Compiler:
    debug_command: List[str]
    release_command: List[str]


class Problem:

    # def generate_input()
    generate_input: Callable[[], None] = None

    # def checker(icontent: str, ocontent: str, econtent: str) -> Optional[str]
    checker: Callable[[str, str, str], str] = None

    @staticmethod
    def call_generate_input():
        if Problem.generate_input is None:
            raise ProblemError(Problem.generate_input_error)
        sys.stdout.flush()
        f = open(INPUT_FILE, "w")
        fd_file = f.fileno()
        fd_out = sys.stdout.fileno()
        fd_out2 = os.dup(fd_out)
        os.dup2(fd_file, fd_out)
        Problem.generate_input()
        sys.stdout.flush()
        os.close(fd_out)
        f.close()
        os.dup2(fd_out2, fd_out)
        os.close(fd_out2)
    
    @staticmethod
    def call_checker(input_file: Path, output_file: Path, expected_file: Path) -> Optional[str]:
        if Problem.checker is None:
            raise ProblemError(Problem.checker_error)
        with open(input_file, "r") as f:
            in_content = f.read()
        with open(output_file, "r") as f:
            out_content = f.read()
        if output_file == expected_file:
            out_correct_content = out_content
        else:
            with open(expected_file, "r") as f:
                out_correct_content = f.read()
        return Problem.checker(in_content, out_content, out_correct_content)
    
    @staticmethod
    def import_function(name: str, default=None):
        assert name in Problem.__annotations__
        setattr(Problem, f"{name}_error", None)
        try:
            module = importlib.import_module("problem")
        except Exception as e:
            if default is None:
                setattr(Problem, f"{name}_error", str(e))
            else:
                setattr(Problem, name, default)
            return
        if not hasattr(module, name):
            if default is None:
                setattr(Problem, f"{name}_error", "not found")
            else:
                setattr(Problem, name, default)
            return
        pcount = len(Problem.__annotations__[name].__args__) - 1
        func = getattr(module, name)
        if not callable(func):
            setattr(Problem, f"{name}_error", "not callable")
            return
        params = inspect.signature(func).parameters
        if len(params) != pcount:
            setattr(Problem, f"{name}_error", "invalid signature")
            return
        for _, p in params.items():
            if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
                setattr(Problem, f"{name}_error", "invalid signature")
                return
        setattr(Problem, name, func)

    @staticmethod
    def import_all():
        Problem.import_function("generate_input")
        Problem.import_function("checker", default=default_checker)    


class Settings:

    instance: Optional["Settings"] = None

    TABLE = "settings"

    @staticmethod
    def create_instance(*args, **kwargs):
        assert Settings.instance is None
        Settings.instance = Settings(*args, **kwargs)

    def __init__(self):
        exist = SETTINGS_FILE.exists()
        self.connection = sqlite3.connect(str(SETTINGS_FILE), isolation_level="EXCLUSIVE")
        self.connection.row_factory = sqlite3.Row
        if not exist:
            cursor = self.connection.execute(f"CREATE TABLE {Settings.TABLE} (key TEXT, value TEXT)")
            cursor.close()

    def set_problem(self, problem: str):
        self.update("problem", problem)

    def unset_problem(self):
        self.unset("problem")

    def get_problem(self) -> str:
        return self.get("problem")
    
    def set_contest(self, contest_id: int):
        self.update("contest", str(contest_id))

    def get_contest(self) -> Optional[int]:
        c = self.get("contest")
        if c is None:
            return None
        return int(c)
    
    def set_api(self, key: str, secret: str):
        self.update("api", json.dumps({"key": key, "secret": secret}))

    def get_api(self) -> Optional[Tuple[str, str]]:
        js_str = self.get("api")
        if js_str is None:
            return None
        js = json.loads(js_str)
        return js["key"], js["secret"]
    
    def set_auth(self, login: str, password: str):
        self.update("auth", json.dumps({"login": login, "password": password}))
    
    def get_auth(self) -> Optional[Dict[str, str]]:
        js_str = self.get("auth")
        if js_str is None:
            return None
        return json.loads(js_str)
    
    def get_login(self) -> Optional[str]:
        auth = self.get_auth()
        if auth is None:
            return None
        return auth["login"]
    
    def set_compiler(self, compiler: Compiler):
        self.update("compiler", json.dumps({"debug": compiler.debug_command, "release": compiler.release_command}))
    
    def get_compiler(self) -> Optional[Compiler]:
        js_str = self.get("compiler")
        if js_str is None:
            return None
        js = json.loads(js_str)
        return Compiler(js["debug"], js["release"])
    
    def set_codeforces_compiler(self, compiler: CodeforcesCompiler):
        self.update("codeforces_compiler", compiler.value)

    def get_codeforces_compiler(self, required=False) -> Optional[CodeforcesCompiler]:
        value = self.get("codeforces_compiler")
        if value is None and required:
            raise Error("Codeforces compiler not set")
        return CodeforcesCompiler(value)
    
    def set_exe_info(self, src_timestamp: int, release: bool):
        self.update("exe_info", json.dumps({"src_timestamp": src_timestamp, "release": release}))

    def get_exe_info(self) -> Optional[Tuple[int, bool]]:
        js_str = self.get("exe_info")
        if js_str is None:
            return None
        js = json.loads(js_str)
        return js["src_timestamp"], js["release"]
    
    def get(self, key: str) -> Optional[str]:
        cursor = self.connection.execute(f"SELECT value FROM {Settings.TABLE} WHERE key=?", [key])
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        return row["value"]

    def update(self, key: str, value: str):
        if self.get(key) is not None:
            cursor = self.connection.execute(f"UPDATE {Settings.TABLE} SET value=? WHERE key=?", [value, key])
        else:
            cursor = self.connection.execute(f"INSERT INTO {Settings.TABLE} VALUES (?, ?)", [key, value])
        cursor.close()
        self.connection.commit()

    def unset(self, key: str):
        cursor = self.connection.execute(f"DELETE FROM {Settings.TABLE} WHERE key=?", [key])
        cursor.close()
        self.connection.commit()
    

class Api:

    instance: Optional["Api"] = None

    URL = "https://codeforces.com/api"
    WAIT_INTERVAL = timedelta(seconds=3)

    @staticmethod
    def create_instance(*args, **kwargs):
        assert(Api.instance is None)
        Api.instance = Api(*args, **kwargs)    
    
    def __init__(self, key: str, secret: str):
        self.last_request_time = datetime(year=2000, month=1, day=1, tzinfo=timezone.utc)
        self.key = key
        self.secret = secret

    def request_actual_contests(self) -> List[Contest]:
        all_contests, _ = self.request("contest.list", {"gym": "false"})
        not_finished = filter(lambda e: e["phase"] in ("BEFORE", "CODING"), all_contests)
        return list(Contest(e["name"], e["id"], datetime.fromtimestamp(e["startTimeSeconds"])) for e in not_finished)

    def request_last_submission(self, login: str, contest: int) -> Submission:
        response, _ = self.request("contest.status", {"contestId": contest, "handle": login, "from": "1", "count": "1"})
        sub = response[0]
        result = Submission(sub["contestId"], sub["problem"]["index"], sub["passedTestCount"], sub["creationTimeSeconds"])
        if "verdict" in sub:
            result.verdict = sub["verdict"]
            result.time_ms = sub["timeConsumedMillis"]
            result.mem_mb = sub["memoryConsumedBytes"] // 1024 // 1024
        return result

    def request_rank(self, contest_id: int, login: str) -> int:
        params = {"contestId": str(contest_id), "handles": f"{login};", "showUnofficial": "true"}
        r, prepared = self.request("contest.standings", params)
        if len(r["rows"]) == 0:
            raise RequestError("Standings, len(rows) == 0", prepared)
        return int(r["rows"][0]["rank"])

    def request_users(self, contest_id: int) -> List[User]:
        params = {"contestId": str(contest_id), "activeOnly": "false", "includeRetired": "false"}
        r, prepared = self.request("user.ratedList", params)
        if len(r) == 0:
            raise RequestError("Users, len == 0", prepared)
        return list(User(e["handle"], e["rating"], e["maxRating"]) for e in r)

    def request_standings(self, contest_id: int, from_place: int, count: int) -> List[Contestant]:
        params = {"contestId": str(contest_id), "showUnofficial": "true", "from": str(from_place), "count": str(count)}
        r, prepared = self.request("contest.standings", params)
        if len(r["rows"]) == 0:
            raise RequestError("Standings, len(rows) == 0", prepared)
        return list(Contestant(e["party"]["members"][0]["handle"], e["rank"]) for e in r["rows"])
        
    def request(self, method_name: str, params: Dict[str, str]) -> Tuple[Any, requests.PreparedRequest]:
        next_request_time = self.last_request_time + Api.WAIT_INTERVAL
        now = datetime.now(timezone.utc)
        if now < next_request_time:
            time.sleep((next_request_time - now).total_seconds())
        params = self.calc_request_params(method_name, params)
        try:
            response = send_request("GET", f"{Api.URL}/{method_name}", params=params)
        finally:
            self.last_request_time = datetime.now(timezone.utc)
        result = json.loads(response.text)
        if result["status"] != "OK":
            raise RequestError("API response status is not OK", response)
        return result["result"], response.request
    
    def calc_request_params(self, method_name: str, params: Dict[str, str]) -> dict:
        params = params.copy()
        params["lang"] = "en"
        params["apiKey"] = self.key
        params["time"] = str(int(time.time()))
        sorted_params = sorted(params.items())
        rnd = "".join(random.choices(string.ascii_lowercase, k=6))
        sigstr = f"{rnd}/{method_name}?{sorted_params[0][0]}={sorted_params[0][1]}"
        sigstr = functools.reduce(lambda r, e: f"{r}&{e[0]}={e[1]}", sorted_params[1:], sigstr)
        sigstr = f"{sigstr}#{self.secret}"
        sig = rnd + hashlib.sha512(sigstr.encode("utf-8"), usedforsecurity=True).hexdigest()
        params["apiSig"] = sig
        return params
    

def send_request(method: str, url: str, *, params: dict = None, data: dict = None, session: requests.Session = None, timeout=None) -> requests.Response:
    assert method in ("GET", "POST")
    if need_close := session is None:
        session = requests.Session()
    if timeout is None:
        timeout = TIMEOUT_DEFAULT
    user_agent = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"}
    request = requests.Request(method, url, params=params, data=data, headers=user_agent)
    prepared = session.prepare_request(request)
    try:
        response = session.send(prepared, timeout=timeout)
    except requests.RequestException as e:
        raise RequestError(str(e), prepared) from None
    finally:
        if need_close:
            session.close()
    if response.status_code != 200:
        raise RequestError("Bad response status", response)
    return response
    

def browser_send_request(request: dict) -> Any:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("localhost", BROWSER_PORT))
    except ConnectionError as e:
        raise Error("Cannot connect to the browser" + " " + str(e))
    data = pickle.dumps(request)
    sock.send(len(data).to_bytes(length=4, byteorder="little"))
    sock.send(data)
    length = int.from_bytes(sock.recv(4), byteorder="little")
    data = sock.recv(length)
    sock.close()
    response = pickle.loads(data)
    if isinstance(response, Exception):
        raise Error(str(response))
    return response


def browser_request_stop_browser() -> int:
    request = {"cmd": "stop_browser"}
    return browser_send_request(request)


def browser_request_participants_count(contest_id: int) -> int:
    request = {"cmd": "participants_count", "contest_id": contest_id}
    return browser_send_request(request)


def browser_request_submit_problem(contest_id: int, problem: str, code: str, cf_compiler: CodeforcesCompiler) -> int:
    request = {"cmd": "submit_problem", "contest_id": contest_id, "problem": problem, "cf_compiler": cf_compiler.value, "code": code}
    return browser_send_request(request)


def browser_request_sample_test(contest_id: int, problem: str, index: int) -> Tuple[str, str]:
    request = {"cmd": "test_sample", "contest_id": contest_id, "problem": problem, "index": index}
    return browser_send_request(request)


def input_and_test_compiler_cmd(compilation_type: str):
    with open(SOURCE_FILE, "w") as f:
        f.write("#include <iostream> \n int main(){ std::cout << 888 << std::endl; return 0; } \n ")
    cmd = input(f"Compilation command ({compilation_type}): ")
    cmd = cmd.strip().split(" ")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        out_txt = subprocess.run([EXECUTABLE_FILE], check=True, capture_output=True).stdout.decode("utf-8")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise Error(e) from None
    if "888" not in out_txt:
        raise Error("Compiler failed test")
    return cmd


def print_c(color: str, *args, **kwargs):
    assert color in ("green", "red", "blue", "cyan", "yellow", "white", "magenta")
    color = getattr(colorama.Fore, color.upper())
    print(color, colorama.Style.BRIGHT, sep="", end="")
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end="")


def default_checker(_i: str, out_content: str, expected_content: str) -> Optional[str]:
    out_strings = [s.rstrip() for s in out_content.split("\n")]
    del out_content
    expected_strings = [s.rstrip() for s in expected_content.split("\n")]
    del expected_content
    len_min = min(len(out_strings), len(expected_strings))
    len_max = max(len(out_strings), len(expected_strings))
    max_strings = out_strings if len_max == len(out_strings) else expected_strings
    for i in range(len_min):
        if out_strings[i] != expected_strings[i]:
            return f"error line: {i + 1}"
    for i in range(len_min, len_max):
        if len(max_strings[i]) != 0:
            return f"extra line non-empty: {i + 1}"
    return None


def set_stack_limit(mbytes: int):
    if "win" in platform.platform().lower():
        return
    resource = importlib.import_module("resource")
    need = mbytes * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft >= need:
        return
    value = need if hard == resource.RLIM_INFINITY else min(hard, need)
    resource.setrlimit(resource.RLIMIT_STACK, (value, hard))


def execute(correct_solution=False):
    set_stack_limit(256)
    cmd = [str(EXECUTABLE_FILE)]
    if correct_solution:
        cmd.append("--correct-solution")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise Error(e) from None
    except subprocess.CalledProcessError:
        raise ExecutionError() from None
    

def compare(need_exec=True) -> Optional[str]:
    try:
        if need_exec:
            execute()
    except ExecutionError:
        return "Runtime error"
    return Problem.call_checker(INPUT_FILE, OUTPUT_FILE, EXPECTED_FILE)


def work_compile(args):
    compiler = Settings.instance.get_compiler()
    if compiler is None:
        raise Error("Compiler not selected")
    src_mtime = int(os.stat(SOURCE_FILE).st_mtime)
    exe_info = Settings.instance.get_exe_info()
    if args.build or not EXECUTABLE_FILE.exists() or exe_info is None or exe_info[0] != src_mtime or (args.release and not exe_info[1]):
        command = compiler.release_command if args.release else compiler.debug_command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            raise CompilationError(functools.reduce(lambda r, e: f"{r} {e}", command[1:], command[0])) from None
        except FileNotFoundError:
            raise Error(f"Compiler executable not found: {command[0]}") from None
        Settings.instance.set_exe_info(src_mtime, args.release)


def work_execute(args):
    begin = datetime.now(timezone.utc)
    print("executing...")
    execute(correct_solution=False)
    end = datetime.now(timezone.utc)
    milliseconds = int(1000 * (end - begin).total_seconds())
    print_c("white", f"{milliseconds} ms")
    if args.input:
        with open(INPUT_FILE, "r") as f:
            content = f.read().split("\n")
        print_c("white", "Input:")
        for line in content:
            print(" " * 4, line, sep="")
    if args.output:
        with open(OUTPUT_FILE, "r") as f:
            content = f.read().split("\n")
        print_c("white", "Output:")
        for line in content:
            print(" " * 4, line, sep="")


def work_compare():
    print("executing...")
    execute()
    err = compare(need_exec=False)
    if err:
        print_c("red", err)
    else:
        print_c("green", "OK!")

    
def work_compare_solution():
    execute(correct_solution=True)
    os.rename(OUTPUT_FILE, EXPECTED_FILE)
    work_compare()


def work_gendbg_input(args):
    Problem.call_generate_input()
    work_execute(args)


def work_gendbg_runtime(args):
    start = datetime.now(timezone.utc)
    for counter in range(1, 2 ** 20):
        print_c("white", counter, end=" " if counter % 25 > 0 else "\n", flush=True)
        Problem.call_generate_input()
        try:
            execute()
        except ExecutionError:
            print_c("green", "\nTest generated!")
            return
        if counter == 250:
            per_test_ms = 1000 * (datetime.now(timezone.utc) - start).total_seconds() / counter
            print_c("blue", f"\n Per test: {per_test_ms} ms \n")


def work_gendbg_checker(args):
    if Problem.checker is default_checker:
        raise ProblemError("problem.checker - not defined")
    start = datetime.now(timezone.utc)
    for counter in range(1, 2 ** 20):
        print_c("white", counter, end=" " if counter % 25 > 0 else "\n", flush=True)
        Problem.call_generate_input()
        execute(correct_solution=False)
        err = Problem.call_checker(INPUT_FILE, OUTPUT_FILE, OUTPUT_FILE)
        if err:
            print("\n", err, sep="")
            print_c("green", "Test generated!")
            return
        if counter == 250:
            per_test_ms = 1000 * (datetime.now(timezone.utc) - start).total_seconds() / counter
            print_c("blue", f"\n Per test: {per_test_ms} ms \n")


def work_gendbg_solution(args):
    start = datetime.now(timezone.utc)
    for counter in range(1, 2 ** 20):
        print_c("white", counter, end=" " if counter % 25 > 0 else "\n", flush=True)
        Problem.call_generate_input()
        execute(correct_solution=True)
        os.rename(OUTPUT_FILE, EXPECTED_FILE)
        err = compare()
        if err:
            print()
            print_c("white", "Error:", err)
            print_c("green", "Test generated!")
            return
        if counter == 250:
            per_test_ms = 1000 * (datetime.now(timezone.utc) - start).total_seconds() / counter
            print_c("blue", f"\n Per test: {per_test_ms} ms \n")


def work_request_sample(args):
    contest = Settings.instance.get_contest()
    if contest is None:
        raise Error("Contest not selected")
    problem = Settings.instance.get_problem()
    if problem is None:
        raise Error("Problem not selected")
    sample = browser_request_sample_test(contest, problem, args.sample - 1)
    with open(INPUT_FILE, "w") as f:
        f.write(sample[0])
    with open(EXPECTED_FILE, "w") as f:
        f.write(sample[1])


def work_submit_solution(args):
    if Api.instance is None:
        raise Error("API not set")
    contest = Settings.instance.get_contest()
    if contest is None:
        raise Error("Contest not selected")
    problem = Settings.instance.get_problem()
    if problem is None:
        raise Error("Problem not selected")
    cf_compiler = Settings.instance.get_codeforces_compiler()
    if cf_compiler is None:
        raise Error("Codeforces compiler not set")
    login = Settings.instance.get_login()
    if login is None:
        raise Error("Login not set")
    with open(SOURCE_FILE, "r") as f:
        code = f.read()
    request_time = datetime.now(timezone.utc)
    print_c("white", f"{contest}{problem} submitting...")
    browser_request_submit_problem(contest, problem, code, cf_compiler)
    print("solution submitted")
    while True:
        sub = Api.instance.request_last_submission(login, contest)
        sub_time = datetime.fromtimestamp(sub.creation_timestamp, timezone.utc)
        if sub_time < request_time:
            print("Getting submission status...")
        else:
            break
    prev_tests_passed = -1
    while True:
        sub = Api.instance.request_last_submission(login, contest)
        if sub.verdict is None or sub.verdict == "TESTING":
            if sub.tests_passed != prev_tests_passed:
                prev_tests_passed = sub.tests_passed
                print_c("yellow", f"Testing... {sub.tests_passed + 1}")
            continue
        break
    if sub.verdict != "OK":
        print_c("red", sub.verdict)
        return
    Settings.instance.unset_problem()
    print_c("white", "Time:", sub.time_ms, "ms")
    print_c("white", "Mem:", sub.mem_mb, "mb")
    print_c("green", "Solution accepted!")  


def work_choose_contest(args):
    print_c("white", "Getting actual contests list...")
    if Api.instance is None:
        raise Error("API not set")
    contests = Api.instance.request_actual_contests()
    if len(contests) == 0:
        print_c("white", "No actual contests")
        return
    for i, c in enumerate(contests):
        print_c("white", f"{i}.", end=" ")
        print(c.name)
    while True:
        try:
            index = int(input("Index: "))
        except ValueError:
            print("Wrong value")
        else:
            if index < 0 or index >= len(contests):
                print("Wrong value")
            else:
                break
    args.contest = contests[index].id
    work_set_contest(args)


def work_set_contest(args):
    if args.contest == 0:
        work_choose_contest(args)
        return
    Settings.instance.set_contest(args.contest)
    Settings.instance.unset_problem()
    print_c("blue", f"Contest selected {args.contest}")
    print_c("white", f"Problem not selected")


def work_set_problem(args):
    regexp = re.compile(r"[A-Z][0-9]{0,1}")
    p = args.problem.upper()
    if regexp.fullmatch(p) is None:
        raise Error("Invalid problem name")
    Settings.instance.set_problem(p)
    print_c("blue", f"Problem selected {p}")


def work_rank(args):
    if Api.instance is None:
        raise Error("API not set")
    contest = Settings.instance.get_contest()
    if contest is None:
        raise Error("Contest not selected")
    login = Settings.instance.get_login() if args.rank == "#" else args.rank
    if login is None:
        raise Error("Login not set")
    pcount = browser_request_participants_count(Settings.instance.get_contest())
    rank = Api.instance.request_rank(contest, login)
    percent = 100.0 * rank / pcount
    color_map = [(1.0, "red"), (3.0, "yellow"), (6.0, "magenta"), (15.0, "blue"), (27.0, "cyan"), (45.0, "green"), (101.0, "white")]
    for p, c in color_map:
        if percent <= p:
            color = c
            break
    print_c("white", f"{rank}", end=" / ")
    print_c(color, f"{percent:.2f}%")
    print_c("white", f"{pcount}")


def work_predict(args):
    if Api.instance is None:
        raise Error("API not set")
    contest = Settings.instance.get_contest()
    if contest is None:
        raise Error("Contest not set")
    login = Settings.instance.get_login()
    if login is None:
        raise Error("Login not set")
    rating_map = {e.handle: e.rating for e in Api.instance.request_users(contest)}
    login_place = Api.instance.request_rank(contest, login)
    near_contestants = Api.instance.request_standings(contest, login_place - 100, 200)
    rating_sum, used = 0, 0
    for contestant in near_contestants:
        if contestant.handle in rating_map:
            rating_sum += rating_map[contestant.handle]
            used += 1
    perfomance = rating_sum // used
    delta = (perfomance - rating_map[login]) // 4
    color = [(1199, "white"), (1399, "green"), (1599, "cyan"), (1899, "blue"), (2099, "magenta"), (2399, "yellow"), (5000, "red")]
    for r, c in color:
        if perfomance <= r:
            print_c("white", "Perfomance:", end=" ")
            print_c(c, perfomance)
            break
    delta_color = "green" if delta > 0 else ("red" if delta < 0 else "white")
    print_c("white", "Delta:", end=" ")
    if delta > 0:
        delta = "+" + str(delta)
    print_c(delta_color, delta)
    

def work_set_api(args):
    key = input("API key: ")
    secret = input("API secret: ")
    Settings.instance.set_api(key, secret)
    print_c("blue", "API settings saved")


def work_set_auth(args):
    login = input("Login: ")
    password = input("Password: ")
    Settings.instance.set_auth(login, password)
    print_c("blue", "Auth settings saved")


def work_set_compiler(args):
    print("Command must compile main.cpp into main.exe")
    print("For example: g++ -std=c++17 -ggdb3 -Wall -o main.exe main.cpp")
    if SOURCE_FILE.exists() and not SOURCE_FILE_BACKUP.exists():
        os.rename(SOURCE_FILE, SOURCE_FILE_BACKUP)
    debug_cmd = input_and_test_compiler_cmd("debug")
    release_cmd = input_and_test_compiler_cmd("release")
    compiler = Compiler(debug_cmd, release_cmd)
    if SOURCE_FILE_BACKUP.exists():
        os.rename(SOURCE_FILE_BACKUP, SOURCE_FILE)
    Settings.instance.set_compiler(compiler)
    print_c("green", "OK")


def work_set_codeforces_compiler(args):
    compilers = list(CodeforcesCompiler)
    for i, c in enumerate(compilers):
        print_c("white", f"{i}.", end=" ")
        print(c.name)
    while True:
        try:
            index = int(input("Index: "))
        except ValueError:
            print("Wrong value")
        else:
            if index < 0 or index >= len(compilers):
                print("Wrong value")
            else:
                break
    c = compilers[index]
    Settings.instance.set_codeforces_compiler(c)
    print_c("blue", f"Compiler: {c.name}")


def work_show_settings(args):
    s = Settings.instance
    print("Contest:", end=" ")
    print_c("blue", s.get_contest())
    print("Problem:", end=" ")
    print_c("blue", s.get_problem())
    cf_compiler = s.get_codeforces_compiler()
    print("CF compiler:", cf_compiler.name if cf_compiler else None)
    print("User:", s.get_login())
    print("API:", ("OK" if s.get_api() else None))
    print("Auth:", ("OK" if s.get_auth() else None))
    compiler = Settings.instance.get_compiler()
    if compiler:
        print("Compile (debug):", functools.reduce(lambda r, e: f"{r} {e}", compiler.debug_command, ""))
        print("Compile (release):", functools.reduce(lambda r, e: f"{r} {e}", compiler.release_command, ""))


def work_start_browser(args):
    auth = Settings.instance.get_auth()
    if auth is None:
        raise Error("Auth not set")
    login, password = auth.values()
    subprocess.run([sys.executable, "-m", "venv", "--copies", "./browser_venv"], check=True, capture_output=True)
    python = str(Path("./browser_venv/bin/python").resolve().absolute())
    subprocess.run([python, "-m", "pip", "install", "-r", "browser_requirements.txt"], check=True, capture_output=True)
    browser_cmd = [python, "cf_browser.py", "-l", login, "-p", password]
    if args.start_browser.upper() == "W":
        browser_cmd.append("-w")
    subprocess.Popen(browser_cmd, cwd=".", close_fds=True, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print_c("white", "Started")


def work_stop_browser(args):
    try:
        browser_request_stop_browser()
    except ConnectionError:
        print_c("white", "Not running")
    else:
        print_c("white", "Stopped")

def do_work(args):
    Problem.import_all()
    Settings.create_instance()
    if api := Settings.instance.get_api():
        Api.create_instance(api[0], api[1])
    if args.start_browser is not None:
        work_start_browser(args)
        return
    if args.stop_browser:
        work_stop_browser(args)
        return
    if args.settings:
        work_show_settings(args)
        return
    if args.set_compiler:
        work_set_compiler(args)
        return
    if args.set_auth:
        work_set_auth(args)
        return
    if args.set_api:
        work_set_api(args)
        return
    if args.set_codeforces_compiler:
        work_set_codeforces_compiler(args)
        return
    if args.contest is not None:
        work_set_contest(args)
        return
    if args.rank is not None:
        work_rank(args)
        return
    if args.predict:
        work_predict(args)
        return
    if args.problem is not None:
        work_set_problem(args)
    if args.submit:
        work_submit_solution(args)
        return
    if args.sample is not None:
        work_request_sample(args)
    gendbg = args.gendbg_input or args.gendbg_runtime or args.gendbg_checker or args.gendbg_solution
    if len(sys.argv) == 1 or args.build or args.execute or args.compare or args.compare_solution or gendbg:
        work_compile(args)
    if args.execute:
        work_execute(args)
        return
    if args.compare:
        work_compare()
        return
    if args.compare_solution:
        work_compare_solution()
        return
    if args.gendbg_input:
        work_gendbg_input(args)
        return
    if args.gendbg_runtime:
        work_gendbg_runtime(args)
        return
    if args.gendbg_checker:
        work_gendbg_checker(args)
        return
    if args.gendbg_solution:
        work_gendbg_solution(args)
        return


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser("cf.py", formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35))
    argparser.add_argument("-b", "--build", action="store_true", help="Build executable")
    argparser.add_argument("-r", "--release", action="store_true", help="Build release")
    argparser.add_argument("-e", "--execute", action="store_true", help="Run code")
    argparser.add_argument("-c", "--compare", action="store_true", help="Compare output with correct one (call checker)")
    argparser.add_argument("-i", "--input", action="store_true", help="Show input file on execution")
    argparser.add_argument("-o", "--output", action="store_true", help="Show output after execution")
    argparser.add_argument("-p", "--problem", type=str, metavar="P", help="Select problem")
    argparser.add_argument("-s", "--submit", action="store_true", help="Submit current problem")
    argparser.add_argument("-t", "--sample", type=int, metavar="I", help="Get test sample")
    argparser.add_argument("-q", "--rank", type=str, nargs="?", default=None, const="#", metavar="USER", help="Show rank for current contest")
    argparser.add_argument("-w", "--settings", action="store_true", help="Show current settings")
    argparser.add_argument("--compare-solution", action="store_true", help="Compare solution with correct one on current input")
    argparser.add_argument("--gendbg-input", action="store_true", help="Generate input and run once")
    argparser.add_argument("--gendbg-runtime", action="store_true", help="Test solution for runtime error")
    argparser.add_argument("--gendbg-checker", action="store_true", help="Test solution with a checker")
    argparser.add_argument("--gendbg-solution", action="store_true", help="Test solution with correct one")
    argparser.add_argument("--contest", type=int, nargs="?", default=None, const=0, metavar="ID", help="Select contest")
    argparser.add_argument("--predict", action="store_true", help="Predict rating delta")
    argparser.add_argument("--set-auth", action="store_true", help="Set login/password")
    argparser.add_argument("--set-api", action="store_true", help="Set API key/secret")
    argparser.add_argument("--set-compiler", action="store_true", help="Set compilation command")
    argparser.add_argument("--set-codeforces-compiler", action="store_true", help="Set compiler for submissions")
    argparser.add_argument("--start-browser", type=str, nargs="?", default=None, const="#", help="Start browser service (W for window)")
    argparser.add_argument("--stop-browser", action="store_true", help="Stop browser service")
    return argparser.parse_args()
    

def main():
    args = parse_args()
    os.chdir(ROOT)
    try:
        do_work(args)
    except KeyboardInterrupt:
        print()
        return
    except CompilationError as e:
        print("Compilation string:", e.compilation_string)
        return
    except ProblemError as e:
        print()
        print_c("red", "problem.py error")
        print(e)
        return
    except RequestError as e:
        print_c("red", "request failed")
        print(e)
        print("Request:", e.request.method, e.request.url)
        if e.response is not None:
            print("Status:", e.response.status_code)
        return
    except Error as e:
        print(e)
        return


if __name__ == "__main__":
    main()
