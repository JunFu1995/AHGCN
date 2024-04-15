import subprocess

tpl = 'nohup python main_cviqd_final.py --resume live2depoch_0058.pth --k %d --use_hgcn %d --use_ms %d --gpu %d --use_fc %d > cviqd_use_hgcn_%d_use_ms_%d_k_%d_use_fc_%d.log &'


gpu = 0
use_hgcn = 1
use_ms = 1
use_fc = 0
for gpu, k in zip([0], [5]):
    print(tpl % (k,use_hgcn,use_ms,gpu, use_fc, use_hgcn,use_ms,k, use_fc))
    subprocess.call(tpl % (k,use_hgcn,use_ms,gpu, use_fc, use_hgcn,use_ms,k, use_fc), shell=True)