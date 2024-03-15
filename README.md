## cf-assist

cf-assist is a single file python script that makes easier your participation in codeforces contests

```
$ python3 cf.py -h
options:
  -h, --help                 show this help message and exit
  -b, --build                Build executable
  -r, --release              Build release
  -e, --execute              Run code
  -c, --compare              Compare output with correct one
  -p P, --problem P          Select problem
  -s, --submit               Submit current problem
  -t I, --sample I           Get test sample
  -q, --rank                 Show rank for current contest
  -w, --settings             Show current settings
  --compare-solution         Compare solution with correct one on current input
  --generate                 Try generate bad test
  --contest [ID]             Select contest
  --check-auth               Check authorization
  --set-auth                 Authorize at codeforces
  --set-api                  Set API key/secret
  --set-compiler             Set compilation command
  --set-codeforces-compiler  Set compiler for submissions
```

## How to use

For solutions only C++ is supported

It is inteded that you use following file structure

main.exe should read from input.txt and write into output.txt

```
ROOT
|-- main.cpp       solution source
|-- cf.py          script
|-- main.exe       solution executable
|-- input.txt      solution input
|-- output.txt     solution output
|-- expected.txt   correct output
|-- problem.py     problem input generator/checker
|-- cf.settings    settings file (sqlite3)
```

First of all configure your settings

You can create api key [here](https://codeforces.com/settings/api)

```
python3 cf.py --set-compiler
python3 cf.py --set-codeforces-compiler
python3 cf.py --set-auth
python3 cf.py --set-api
```

Next, select contest where you plan to participate

```
python3 cf.py --contest
```

Select problem A
```
python3 cf.py -p A
```

Download first input/output for this problem into input.txt/expected.txt
```
python3 cf.py -t 1
```

After you write a solution, check that your output.txt is the same as expected.txt

If executable is out of date, solution is recompiled automatically

```
python3 cf.py -c
```

If output is correct, submit problem A
```
python3 cf.py -s
```

Select the next problem and download test...
```
python3 cf.py -p B -t 1
```

If you get request errors it is more likely codeforces gives you a captcha

So in this case you should copy/paste tests and submit by hands😂

<br>

Also if you cannot find an error in your code and you have a correct solution

It is possible to test your solution on generated tests

For this you need to write an input generator for a problem

```
# file: problem.py

def generate_input(print: Callable):
    print(1, 2, 3) # goes into input.txt

```

And paste correct solution into your code like this

```
void correct_solution() { ... }

int main(int argc, char** argv)
{
    bool CORRECT_SOLUTION = (argc >= 2) && (strcmp(argv[1], "--correct-solution") == 0);

    if (CORRECT_SOLUTION)
        correct_solution();
    else
        my_solution();
}
```

If there are multiple correct outputs, you can write a checker

It should return an error text or None

```
# file: problem.py

def checker(in_content: str, out_content: str, expected_content: str) -> Optional[str]
    pass

```