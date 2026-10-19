# 試作: 低音ピアノを主役にした夜の音(オリジナル、既存曲の旋律は使わない)。C#マイナー、84BPM、12小節
import numpy as np, scipy.signal as ss, scipy.io.wavfile as wf
sr=44100; bpm=84; beat=60/bpm; bar=4*beat; NB=82; L=int(sr*(NB*bar+6))
out=np.zeros((L,2))
rng=np.random.default_rng(7)
mf=lambda m:440*2**((m-69)/12)
def piano(m,dur,vel,bright=1.0):
    # 実際のピアノに近づける: 打弦位置(弦長の1/7)による倍音の欠け、低音弦の強い不協和性、
    # 1音3本の弦のわずかな音程差によるうなり、二段階の減衰(打鍵直後の速い減衰+長い余韻)、ハンマーの打撃音、響板の胴鳴り
    f=mf(m); n=int(sr*dur); t=np.arange(n)/sr; s=np.zeros(n)
    B=0.00012 if m<45 else 0.0003; nk=48 if m<45 else 16
    hard=0.6+0.5*vel                                  # 強く弾くほど倍音が多い
    for k in range(1,nk+1):
        fk=f*k*np.sqrt(1+B*k*k)
        if fk>sr/2.3: break
        a=vel*abs(np.sin(np.pi*k/7))/k**(1.25/hard)*(bright**(k-1))
        tau1=1.2/(1+0.18*k)*(110/f)**0.15; tau2=tau1*7
        d=0.75*np.exp(-t/tau1)+0.25*np.exp(-t/tau2)
        for dt in (1.0,1.0007,0.9994):
            s+=a*d*np.sin(2*np.pi*fk*dt*t+rng.uniform(0,2*np.pi))/3
    th=rng.standard_normal(n)*np.exp(-t/0.012)*0.4*vel           # ハンマーの打撃音(低く鈍い)
    s+=ss.lfilter(*ss.butter(2,min(0.9,max(f*6,300)/(sr/2))),th)
    bt=np.arange(int(sr*0.08))/sr; body=rng.standard_normal(len(bt))*np.exp(-bt/0.02)  # 響板の胴鳴り
    body=ss.lfilter(*ss.butter(2,[90/(sr/2),900/(sr/2)],'band'),body); s=s+0.35*ss.fftconvolve(s,body)[:n]/np.abs(body).sum()*8
    s*=np.minimum(1,(dur-t)/0.25).clip(0)
    return s
def sub(m,dur,vel):
    f=mf(m); t=np.arange(int(sr*dur))/sr; return vel*np.sin(2*np.pi*f*t)*np.exp(-t/2.5)*np.minimum(1,t/0.02)
def pad(ms,dur,vel):
    n=int(sr*dur); t=np.arange(n)/sr; L_=np.zeros(n); R_=np.zeros(n)
    for m in ms:
        for dt,ch in ((1.0,0),(1.004,1),(0.997,0),(1.002,1)):
            ph=(mf(m)*dt*t)%1.0; saw=2*ph-1
            (L_ if ch==0 else R_)[:]+=saw
    b,a=ss.butter(2,900/(sr/2)); env=np.minimum(1,t/2.0)*np.minimum(1,(dur-t)/1.5).clip(0)
    return np.stack([ss.lfilter(b,a,L_),ss.lfilter(b,a,R_)],1)*env[:,None]*vel/len(ms)
def put(sig,t0,pan=0.0):
    i=int(t0*sr); sig=sig if sig.ndim==2 else np.stack([sig*(1-pan)/1.0,sig*(1+pan)/1.0],1)*0.5
    j=min(L,i+len(sig)); out[i:j]+=sig[:j-i]

# 10/5版: 映像(223.7秒、組み上げ88.05〜197.1、結び214.9)。元はの区間に合わせて展開する。84BPM、1小節=約2.86秒
progA=[(25,[61,64,68]),(21,[61,64,69]),(30,[61,66,69]),(32,[60,63,68])]   # C#m A F#m G#
progB=[(21,[61,64,69]),(28,[63,68,71]),(23,[63,66,71]),(25,[61,64,68])]   # A E B C#m(組み上げ後半の変化)
def layers(b):
    s=b*bar
    if s<20:  return dict(lv=0.45,pad=0.0,mid=0,arp=0)      # 題名・引用・台座: 低音ピアノだけ
    if s<31:  return dict(lv=0.5,pad=0.05,mid=0,arp=0)      # 指示文: 持続音が入る
    if s<81.65:  return dict(lv=0.55,pad=0.07,mid=1,arp=0)     # 第一稿〜第五稿: 中音の和音
    if s<88.05:  return dict(lv=0.5,pad=0.06,mid=0,arp=0)      # 運び込み: いったん引く
    if s<197.1: return dict(lv=0.6,pad=0.06+0.05*(s-88.05)/109.05,mid=1,arp=1)   # 組み上げ: 分散和音、徐々に厚く
    return dict(lv=0.5,pad=0.04,mid=0,arp=0)                # 結び: 低音だけに戻る
NBAR=int(214.9/bar)
for b in range(NBAR):
    s=b*bar; ly=layers(b); prog=progB if (143.6<=s<197.1 and (b//4)%2==1) else progA
    root,ch=prog[b%4]; t0=s
    put(piano(root,bar+2.5,ly["lv"]),t0,-0.2); put(piano(root+12,bar+2.5,ly["lv"]*0.6),t0,-0.1); put(sub(root+12,bar,0.08),t0)
    if ly["pad"]>0: put(pad([m-12 for m in ch],bar+1.5,ly["pad"]),t0)
    if ly["mid"]: put(piano_mid:=None or piano(ch[0],bar,0.13,0.75),t0+beat,0.25); [put(piano(m,bar,0.13,0.75),t0+beat+0.05*(i+1),0.25) for i,m in enumerate(ch[1:])]
    if ly["arp"]:
        arp=[ch[0],ch[1],ch[2],ch[1]+12]; v=0.04+0.03*min(1,(s-88.05)/109.05)
        for i in range(8): put(piano(arp[i%4]+12,beat*1.5,v,0.6),t0+i*beat/2,0.4*np.sin(i))
    if 66<=s<69:                                            # 第五稿(1:09): 言葉が形になる瞬間に高い和音を一つ
        for i,m in enumerate(ch): put(piano(m+12,4,0.12,0.7),t0+0.03*i,0.3)
put(piano(25,12,0.7),NBAR*bar,-0.2); put(piano(37,12,0.4),NBAR*bar,-0.1)   # 終わりの低いC#を長く
tr=np.arange(int(sr*3.2))/sr
ir=np.stack([rng.standard_normal(len(tr))*np.exp(-tr/0.9) for _ in range(2)],1)*0.02
wet=np.stack([ss.fftconvolve(out[:,c],ir[:,c])[:L] for c in range(2)],1)
mix=out+0.6*wet; mix=mix[:int(sr*227.2)]; mix[-int(sr*1.5):]*=np.linspace(1,0,int(sr*1.5))[:,None]
mix=np.tanh(mix/np.abs(mix).max()*1.6)/np.tanh(1.6)*0.89
wf.write('full.wav',sr,(mix*32767).astype(np.int16)); print(len(mix)/sr, NBAR)
