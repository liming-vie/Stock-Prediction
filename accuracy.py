#!/usr/bin/env python
# encoding: utf-8

import sys

if __name__=='__main__':
  if len(sys.argv) != 4:
    print 'Usage: python proc_result.py result_file number_file output_file'
    sys.exit(1)

  result_file, number_file, output_file=sys.argv[1:]

  print 'Loading number info...'
  number_info = {}
  with open(number_file) as fin:
    for line in fin:
      ps=line.split(' ')
      number_info[ps[1]] = (ps[-3], ps[-2])

  cnt, tot=0, 0 # per stock
  count, total = 0, 0 # all
  result={}
  name = None

  print 'Calculating accuracy...'
  fin=open(result_file)
  fin.readline()
  for line in fin:
    ps = line.split(' ')
    if len(ps) < 6:
      break
    if not name:
      name=ps[-5]
    elif name != ps[-5]:
      result[name]=[cnt, tot, float(cnt)/tot]
      name=ps[-5]
      cnt, tot=0, 0
    if ps[-2]==('1' if float(ps[-3]) > 0 else '0'):
      cnt+=1
      count += 1
    tot+=1
    total += 1
  fin.close()
  result[name]=[cnt, tot, float(cnt)/tot]

  print 'Saving accuracy in file %s'%output_file
  result = sorted(result.items(), key=lambda x: x[1][2], reverse=True)
  with open(output_file, 'w') as fout:
    fout.write("name correct total accuracy news_num prices_num\n")
    fout.write("Total %d %d %f\n"%(count, total, float(count)/total))
    for name, data in result:
      fout.write("%s %d %d %f %s %s\n"%(
        name, data[0], data[1], data[2], number_info[name][0], number_info[name][1]))
