# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:13:02 2018

@author: yuxi
"""

import glob


count = 0
read_files = glob.glob("*.txt")
with open("result.txt", "w") as outfile:
    for f in read_files:
        with open(f, "r") as infile:
            outfile.write(infile.read())
            outfile.write('\n')
            infile.close
            count = count + 1