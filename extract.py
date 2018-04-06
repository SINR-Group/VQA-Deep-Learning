import os
import random
import numpy as np

seqSize = 10
dirs = os.listdir('directory path')
count1 = 0 
count2 = 0
for d in dirs:
    files = os.listdir('file path'):
        totalFiles = len(files)
        transFrameIndex = getTransFrameIndex()
        startTransSeq = transFrameIndex - random.randint(3,7)
        if startTransSeq < 0:
            startTransSeq = 0
        endTransSeq = min(startTransSeq + seqSize, totalFiles)
        transSeq = (startTransSeq, endTransSeq)
        count1 += 1
        startNonTransSeq = None
        endNonTransSeq = None
        if totalFiles > 2*seqSize:
            if endTransSeq <= totalFiles/2:
                count2 += 1
                startNonTransSeq = endTransSeq + 1
                endNonTransSeq = startNonTransSeq + seqSize
            elif startTransSeq >= totalFiles/2:
                count2 += 1
                startNonTransSeq = 0
                endNonTransSeq = seqSize
        nonTransSeq = (startNonTransSeq, endNonTransSeq)
        seq = (transSeq, nonTransSeq)
print('transCount: ', count1)
print('nonTransCount: ', count2)
