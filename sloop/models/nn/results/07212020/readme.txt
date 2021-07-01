This is just organized results obtained on 07-19-2020


To reproduce results, open `model/nn/make_shell_reproduce.py`.
Change `dd` to be the path to this directory. Then run
```
python make_shell_reproduce.py
```
This will generate a shell script file called `run_reproduce.sh`.
In this shell script, it contains commands you can run to reproduce any test result.
You could edit that `make_shell_reproduce.py` file if you want to reproduce train/validation results.
