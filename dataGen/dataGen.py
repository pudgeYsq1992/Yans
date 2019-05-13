# -*- coding: utf-8 -*-
import xlrd
import xlwt
import os
import operator
import string
from string import Template
from xlutils.copy import copy
import re
import random

def writeColName(worksheet):
    worksheet.write(1,1,'工作地点')
    worksheet.write(1,2,'工作内容')
    worksheet.write(1,3,'工作时长')
    worksheet.write(1,4,'薪资待遇')
    worksheet.write(1,5,'工作性质为劳务派遣（外包）？')
    worksheet.write(1,6,'公司位置')
    worksheet.write(1,7,'公司规模')
    worksheet.write(1,8,'你有多想去这家公司？')
    return

def genRanddata(worksheet):
    #生成“工作地点”一列
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,1,'上海')
        if randomData == 1:
            worksheet.write(i,1,'南京')
        if randomData == 2:
            worksheet.write(i,1,'日本')
        if randomData == 3:
            worksheet.write(i,1,'其他')
    #生成“工作内容”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,2,'技术研究员')
        if randomData == 1:
            worksheet.write(i,2,'C++程序员')
        if randomData == 2:
            worksheet.write(i,2,'产品经理')
        if randomData == 3:
            worksheet.write(i,2,'其他')
    #生成“工作时长”一列   
    for i in range(2,101):
        randomData = random.randint(0,2)
        if randomData == 0:
            worksheet.write(i,3,'965')
        if randomData == 1:
            worksheet.write(i,3,'995')
        if randomData == 2:
            worksheet.write(i,3,'996')
    #生成“薪资待遇”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,4,'0-10000')
        if randomData == 1:
            worksheet.write(i,4,'10000-15000')
        if randomData == 2:
            worksheet.write(i,4,'15000-25000')
        if randomData == 3:
            worksheet.write(i,4,'25000-30000')
    #生成“工作性质”一列   
    for i in range(2,101):
        randomData = random.randint(0,1)
        if randomData == 0:
            worksheet.write(i,5,'是')
        if randomData == 1:
            worksheet.write(i,5,'否')

    #生成“公司位置”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,6,'CBD')
        if randomData == 1:
            worksheet.write(i,6,'软件园')
        if randomData == 2:
            worksheet.write(i,6,'繁华街区')
        if randomData == 3:
            worksheet.write(i,6,'偏远')
            
    #生成“公司规模”一列   
    for i in range(2,101):
        randomData = random.randint(0,3)
        if randomData == 0:
            worksheet.write(i,7,'上市公司')
        if randomData == 1:
            worksheet.write(i,7,'大公司500人+')
        if randomData == 2:
            worksheet.write(i,7,'小公司50-500人')
        if randomData == 3:
            worksheet.write(i,7,'创业公司')
    return
    

    
def dataGen():
    #data = open('待标注数据.xlsx' ,'w')	
    #data.close()
    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('1')
    writeColName(ws)
    genRanddata(ws)
    #wb.sheets()[0].cell(0,0).value =  'helloworld' 
    wb.save(r'待标注数据.xls')

    return

if __name__ == '__main__':	
    #任务开始
    dataGen()

    print("data generation complete!")