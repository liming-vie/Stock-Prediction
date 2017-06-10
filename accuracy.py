#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import sys

if __name__=='__main__':
  if len(sys.argv) != 5:
    print 'Usage: python proc_result.py result_file number_file fsort_by_acc fsort_by_news'
    sys.exit(1)

  result_file, number_file, output_file, fsort_by_news=sys.argv[1:]

  print 'Loading number info...'
  number_info = {}
  with open(number_file) as fin:
    fin.readline()
    for line in fin:
      ps=line.split(' ')
      number_info[ps[1]] = (ps[-3], ps[-2], float(ps[-1]))

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
    fout.write("name\tcorrect\ttotal\taccuracy\tnews_num\tprices_num\tnews/prices\n")
    fout.write("Total %d\t%d\t%f\n"%(count, total, float(count)/total))
    for name, data in result:
      fout.write("%s\t%d\t%d\t%f\t%s\t%s\t%f\n"%(
        name, data[0], data[1], data[2], \
        number_info[name][0], number_info[name][1],
        number_info[name][2]))

  print 'Saving accuracy sorted by news portion in %s'%fsort_by_news

  result.reverse()
  result = sorted(result, key=lambda x:number_info[x[0]][2])
  with open(fsort_by_news, 'w') as fout:
    fout.write("name\tcorrect\ttotal\taccuracy\tnews_num\tprices_num\tnews/prices\n")
    fout.write("Total %d\t%d\t%f\n"%(count, total, float(count)/total))
    for name, data in result:
      fout.write("%s\t%d\t%d\t%f\t%s\t%s\t%f\n"%(
        name, data[0], data[1], data[2], \
        number_info[name][0], number_info[name][1],
        number_info[name][2]))