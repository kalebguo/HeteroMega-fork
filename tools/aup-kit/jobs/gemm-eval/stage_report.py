#!/usr/bin/env python3
"""Stage durations and a device-time view from one run directory of stages.sh: stage_report.py DIR LABEL..."""
import csv,glob,json,os,statistics as st,sys
ORDER=['start','norm','qkv','rope','kv','dep','sdpa','layout','o','ready','end']
def load(d):
    rows=[]
    for f in glob.glob(d+'/stages_*.csv'):
        fr=[]
        for r in csv.reader(open(f)):
            fr.append((int(r[0]),int(r[1]),int(r[2]),r[3],float(r[4])))
        rows.append(fr)
    return rows
def forwards(fr):
    """Split one file into prefills per chunk; each chunk writes 32*11 rows at its end."""
    out={}
    for c,sl,l,st_,ms in fr:
        out.setdefault(c,[]).append((l,st_,ms))
    res={}
    for c,v in out.items():
        n=32*11
        res[c]=[v[i:i+n] for i in range(0,len(v),n)]
    return res
def stage_table(files,skip=1):
    per={}
    for fr in files:
        fw=forwards(fr)
        for c,lst in fw.items():
            for f in lst[skip:]:
                by={}
                for l,s,ms in f: by.setdefault(l,{})[s]=ms
                for l,m in by.items():
                    for a,b in zip(ORDER,ORDER[1:]):
                        if a in m and b in m: per.setdefault((c,b),[]).append(m[b]-m[a])
                    if l+1 in by and 'end' in m and 'start' in by[l+1]:
                        per.setdefault((c,'gap'),[]).append(by[l+1]['start']-m['end'])
    return per
if __name__=='__main__':
    base=sys.argv[1]
    labels=sys.argv[2:]
    print('%-10s'%'stage'+''.join('%13s'%(x[:12]) for x in labels))
    tab={}
    for lb in labels:
        tab[lb]=stage_table(load(f'{base}/{lb}-stages'))
    for c in (0,1):
        print('chunk',c)
        tot={lb:0 for lb in labels}
        for s in ORDER[1:]+['gap']:
            row='%-10s'%s
            for lb in labels:
                v=tab[lb].get((c,s))
                m=st.median(v) if v else float('nan')
                if v and s!='end' and s!='gap': tot[lb]+=m
                row+='%13.2f'%m
            print(row)
        print('%-10s'%'to ready'+''.join('%13.2f'%tot[lb] for lb in labels))
    print('\nsensors (samples with busy>=30): busy, sclk, fclk, mclk, power, busy CPU cores')
    for lb in labels:
        f=f'{base}/{lb}-sensors.csv'
        if not os.path.exists(f): continue
        rows=[r for r in csv.DictReader(open(f)) if r['busy'] and int(r['busy'])>=30]
        if not rows: continue
        g=lambda k:[float(r[k]) for r in rows if r[k]]
        print('%-12s n=%d busy %.0f sclk %.0f fclk %s mclk %s power %s cores %s'%(lb,len(rows),st.median(g('busy')),st.median(g('sclk')) if g('sclk') else 0,
          st.median(g('fclk')) if g('fclk') else '-',st.median(g('mclk')) if g('mclk') else '-',round(st.median(g('power')),1) if g('power') else '-',round(st.median(g('cores')),1) if g('cores') else '-'))
    print('\nprefill and phases')
    for lb in labels:
        sm=f'{base}/{lb}/summary.csv'
        pre=list(csv.DictReader(open(sm)))[-1] if os.path.exists(sm) else {}
        ph=f'{base}/{lb}-phases.json'
        extra=''
        if os.path.exists(ph):
            d=json.load(open(ph))
            a=d['all']
            extra=' branch %.2f input %.2f run %.2f mutex %.2f gpu_wait %.2f total %.2f serial_gpu %.2f'%(a.get('gpu_branch_ms',0),a['input_wait_ms'],a['run_wait_ms'],a['mutex_wait_ms'],a['gpu_wait_ms'],a['total_ms'],a['serial_gpu_ms'])
        print('%-12s prefill %.1f%s'%(lb,float(pre.get('prefill_median_ms',0) or 0),extra))
