#!/usr/bin/env python3
"""Join device stage times and host phase timers of one run: state fractions per prefill and a text view of a few layers."""
import csv,glob,statistics as st,sys
base,label=sys.argv[1],sys.argv[2]
show=len(sys.argv)>3
stages=list(csv.reader(open(glob.glob(f'{base}/{label}-stages/*.csv')[0])))
timers=list(csv.DictReader(open(glob.glob(f'{base}/{label}-timers/*.csv')[0])))
# Rows of each chunk in order; a forward is 32 layers.
marks={}
for c,sl,l,s,ms in stages:
    marks.setdefault(int(c),[]).append((int(l),s,float(ms)))
n=32*11
fw={c:[v[i:i+n] for i in range(0,len(v),n)] for c,v in marks.items()}
tm={}
for r in timers:
    tm.setdefault(int(r['slot']),[]).append(r)
nf=min(len(v) for v in fw.values())
# Timer slots are numbered by first arrival; choose the chunk of each slot by the spread of host minus device times.
def spread(c,sl,f=3):
    v={};
    for l,s,ms in fw[c][f]: v.setdefault(l,{})[s]=ms
    d=[]
    for l,r in enumerate(tm[sl][f*32:(f+1)*32]):
        h=int(r['start_abs'])/1e6+sum(int(r[p+'_ns'])/1e6 for p in ('setup','prev_wait','gpu_enqueue','input_wait'))
        d.append(h-v[l]['ready'])
    return st.pstdev(d)
if spread(0,0)>spread(0,1):
    tm={0:tm[1],1:tm[0]}
    print('slot 0 is chunk 1')
PH=['setup','prev_wait','gpu_enqueue','input_wait','serial_gpu','d2h','input_scan','mutex_wait','bo_write','bo_sync','run_start','run_wait','y_sync','y_read','output_scan','h2d','gpu_wait','sum']
res=[]
for f in range(1,nf):
    ivals={}
    for c in fw:
        m={}
        for l,s,ms in fw[c][f]: m.setdefault(l,{})[s]=ms
        rows=tm[c][f*32:(f+1)*32]
        offs=[]
        for l,r in enumerate(rows):
            assert int(r['layer'])==l
            t=int(r['start_abs'])/1e6
            cum={}
            for p in PH:
                t+=int(r[p+'_ns'])/1e6
                cum[p]=t
            m[l]['host']=cum
            offs.append(cum['input_wait']-m[l]['ready'])
        ivals[c]=(m,st.median(offs))
    # Common time base: host clock
    base_off=ivals[0][1]
    ev=[]
    A={};B={};N={};W={};Q={}
    for c,(m,off) in ivals.items():
        for l,x in m.items():
            h=x['host']
            A[(c,l)]=(x['start']+off,x['ready']+off)
            bstart=x['ready']+off
            branch=int(tm[c][f*32+l]['gpu_branch_ns'])/1e6
            B[(c,l)]=(bstart,bstart+branch)
            N[(c,l)]=(h['run_start'],h['run_wait'])
            W[(c,l)]=(h['input_wait'],h['y_read'] if h['y_read']>0 else h['run_wait'])
    t0=min(v[0] for v in A.values());t1=max(v[1] for v in B.values())
    # sweep at 0.05 ms resolution over layers 2..29
    a0=A[(0,2)][0];a1=A[(1,29)][1]
    dt=0.05
    tot=0;cnt={}
    t=a0
    while t<a1:
        ga=sum(1 for v in A.values() if v[0]<=t<v[1])
        gb=sum(1 for v in B.values() if v[0]<=t<v[1])
        np_=any(v[0]<=t<v[1] for v in N.values())
        key=(min(ga+gb,2),np_)
        cnt[key]=cnt.get(key,0)+dt
        tot+=dt
        t+=dt
    res.append((cnt,tot,t1-t0,ivals,A,B,N))
agg={}
for cnt,tot,_,_,_,_,_ in res:
    for k,v in cnt.items(): agg[k]=agg.get(k,0)+v/tot/len(res)
print(label,'fraction of time, layers 2-29')
for g in (0,1,2):
    print(' GPU streams active %d: NPU idle %.1f%%  NPU busy %.1f%%'%(g,100*agg.get((g,False),0),100*agg.get((g,True),0)))
print(' whole forward span median %.1f ms'%st.median(r[2] for r in res))
if show:
    cnt,tot,_,ivals,A,B,N=res[len(res)//2]
    t0=A[(0,10)][0]
    print('layers 10-13, ms from chunk 0 layer 10 start; attention start is the end of the layer before')
    rows=[]
    for c in (0,1):
        for l in range(10,14):
            rows.append((A[(c,l)][0]-t0,f'c{c} L{l} attention start  end {A[(c,l)][1]-t0:7.2f}  ({A[(c,l)][1]-A[(c,l)][0]:5.2f})'))
            rows.append((B[(c,l)][0]-t0,f'c{c} L{l} GPU branch start end {B[(c,l)][1]-t0:7.2f}  ({B[(c,l)][1]-B[(c,l)][0]:5.2f})'))
            rows.append((N[(c,l)][0]-t0,f'c{c} L{l} NPU start        end {N[(c,l)][1]-t0:7.2f}  ({N[(c,l)][1]-N[(c,l)][0]:5.2f})'))
    for t,s in sorted(rows):
        print('%8.2f  %s'%(t,s))
