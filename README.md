# GraphLQR

A pytorch implementation of GraphLQR.

Shang Liu, Wanli Gu, Gao Cong, Fuzheng Zhang. Logical Structure Representation Learning with Graph Embedding for Personalized Product Search. In CIKM 2020

**Please cite our CIKM paper if you use our codes. Thanks!**


You can download the CIKMCup 2016 Track 2 Dataset from https://competitions.codalab.org/competitions/11161.

## The requirements are as follows:
	* python>=3.6
	* pytorch>=1.0

## Example to Run
* Make sure the raw data, meta data are in the same direction ./Data/CIKMCup_raw/.
* Preprocessing data. 
   '''
   python utils/1_process.py
   ```

* Start training and test the model. 
   ```
   python run.py
   ```
