"""Fresh-chain native observations with frozen calibration and larger-size tests.

No optimizer updates. Physical source configurations are independent of the
single-pass RefinedWeb corpus that produced the pretrained model states.
"""
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import sentencepiece as spm
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import context_tokens,spin_tokens,prefix_batch,selected_forward
from model_rg.lattice import dyadic_block,observables
from model_rg.provenance import sha256,write_json
from prepare_lattice_data import generate,compile_sampler

ROOT=Path(legacy_path('/pldr-data/model'))
SOURCE=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
OLD=ROOT/'known-universality-20260912'
NAMES=['scripts/fresh_readout_study.py','scripts/prepare_lattice_data.py','scripts/potts_mc.cpp',
       'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/physical_native.py',
       'src/model_rg/lattice.py','src/model_rg/provenance.py']


def sources():return {n:sha256(REPO/n) for n in NAMES}


def native_assets():return {str(p):sha256(p) for p in SOURCE.rglob('*')
                           if p.is_file() and p.suffix in ['.py','.model','.json']}


def initialize(heads,device):
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=TrainingModel(SOURCE,heads,640101,device)
    adapter.model.requires_grad_(False).eval()
    return adapter


@torch.no_grad()
def profile(study,heads,device):
    study=Path(study);study.mkdir(parents=True,exist_ok=True)
    output=study/f'profile-h{heads}.json'
    if output.exists():raise FileExistsError(output)
    adapter=initialize(heads,device)
    processor=spm.SentencePieceProcessor(model_file=str(SOURCE/'tokenizer.model'))
    alphabet=spin_tokens(processor);rows=[]
    rng=np.random.default_rng(218001)
    for q in [2,3]:
        for L in [12,20]:
            x=rng.integers(0,q,(4,L,L),dtype=np.uint8)
            inputs=prefix_batch(x,L*L-1,context_tokens(processor,q,L,1.),alphabet[:q],device)
            capacity=adapter.config.max_position_embeddings
            if inputs.shape[1]>capacity:raise ValueError('Context capacity exceeded')
            torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
            selected,out=selected_forward(adapter,inputs,alphabet[:q],capture=True)
            if not torch.isfinite(selected).all():raise ValueError('Nonfinite profile')
            del out
            full=adapter.forward(inputs[:1]).logits[:,-1,alphabet[:q]]
            torch.testing.assert_close(selected[:1],full,atol=3e-5,rtol=3e-5)
            error=float((selected[:1]-full).abs().max());del full,selected
            adapter.capture=False;adapter.head_outputs={}
            torch.cuda.synchronize(device);peak=torch.cuda.max_memory_allocated(device)/2**30
            if peak>20:raise ValueError('Profile allocation ceiling exceeded')
            rows.append(dict(q=q,L=L,tokens=inputs.shape[1],capacity=capacity,batch=4,
                             peak_gib=peak,seconds=time.perf_counter()-start,max_logit_error=error))
    write_json(output,dict(status='passed',heads=heads,rows=rows,sources=sources(),
        native_assets=native_assets(),native_updates=0,torch=torch.__version__,device=device))
    print(json.dumps(rows),flush=True)


def prepare(study):
    study=Path(study).resolve()
    if not study.is_relative_to(ROOT):raise ValueError('Experiment destination')
    protocol=study/'protocol.json'
    if protocol.exists():raise FileExistsError(protocol)
    profiles={}
    for h in [4,8]:
        p=study/f'profile-h{h}.json';v=json.loads(p.read_text())
        if v['status']!='passed' or v['heads']!=h or v['sources']!=sources() or v['native_assets']!=native_assets():
            raise ValueError('Current larger-prefix profile required')
        profiles[str(h)]=dict(path=str(p),sha256=sha256(p))
    qualification=OLD/'data-qualification/verification.json';v=json.loads(qualification.read_text())
    if v['status']!='passed' or v['sampler_source_sha256']!=sha256(REPO/'scripts/potts_mc.cpp'):
        raise ValueError('Sampler qualification')
    binary=compile_sampler(study)
    if sha256(binary)!=v['sampler_binary_sha256']:raise ValueError('Sampler binary correspondence')
    cases=[]
    for h in [4,8]:
        for origin in ['random','pretrained','adapted']:
            p=None
            if origin=='pretrained':p=ROOT/f'scheduled-training-feasible-20260908/runs/compact-reference1-h{h}-s640101/final-training-state.pt'
            if origin=='adapted':p=OLD/f'confirmation/h{h}-s640101-full/state-16384.pt'
            cases.append(dict(name=f'h{h}-{origin}',heads=h,origin=origin,seed=640101,
                              state=str(p) if p else None,state_sha256=sha256(p) if p else None))
    # Declare the sampling and fitting rule before drawing any configurations.
    design=dict(schema='fresh-readout-v1',status='frozen',sources=sources(),native_assets=native_assets(),
        cases=cases,profiles=profiles,sampler_qualification_sha256=sha256(qualification),
        sizes=[8,12,16,20],q=[2,3],chains=32,samples_per_chain=8,burn=4096,thin=96,
        calibration_chains=list(range(16)),assessment_chains=list(range(16,32)),
        observation='Last proper prefix, all five decoder common rows and final hidden state; first unnormalized query Gram trace. Float32, TF32 disabled, batches of four.',
        fitting=dict(pca_rank=8,ridge=1e-3,normalize='Calibration mean and PCA only; no size-specific transfer normalization',
            within_size='Separate fitted color fractions and direct m2 on first 16 chains of each size and block depth',
            transfer='Freeze a joint L=8,16 depth-zero color-fraction fit; apply without refitting at L=12,20',
            gram='Known token query-norm coefficients and metadata offset; solve zero-sum color system; no fitted target data'),
        bootstrap=dict(replicates=1000,unit='independent Monte Carlo chain',seed=218300,
            fit_uncertainty='Calibration-chain bootstrap with fixed calibration PCA; readout coefficients refitted. PCA acquisition is conditioned on.',
            interval='pointwise percentile; simultaneous claims use Bonferroni paired-chain intervals'),
        blocking='Two sequential 2x2 majority blocks with reproducible independent uniform tie resolution; learned maps use calibration chains only.',
        native_updates=0)
    write_json(study/'design.json',design)
    jobs=[(q,L) for q in [2,3] for L in [8,12,16,20]]
    def draw(item):
        i,(q,L)=item
        c=generate(binary,study/f'data-q{q}-L{L}',q,L,1.,218100+i,32,8,burn=4096,thin=96)
        c['id']=i;return c
    with ThreadPoolExecutor(max_workers=4) as pool:cells=list(pool.map(draw,enumerate(jobs)))
    for n in NAMES:
        p=study/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
    write_json(protocol,dict(**design,cells=cells,design_sha256=sha256(study/'design.json')))
    print('Frozen fresh readout source: 2048 configurations, 256 chains, zero optimizer updates',flush=True)


@torch.no_grad()
def worker(study,heads,device):
    study=Path(study);spec=json.loads((study/'protocol.json').read_text())
    if spec['sources']!=sources() or spec['native_assets']!=native_assets():raise ValueError('Observation sources changed')
    processor=spm.SentencePieceProcessor(model_file=str(SOURCE/'tokenizer.model'));alphabet=spin_tokens(processor)
    for case in spec['cases']:
        if case['heads']!=heads:continue
        out=study/case['name'];out.mkdir(exist_ok=False);start=time.perf_counter()
        adapter=initialize(heads,device)
        if case['state']:
            if sha256(case['state'])!=case['state_sha256']:raise ValueError('Checkpoint changed')
            state=torch.load(case['state'],map_location='cpu',weights_only=False)
            adapter.model.load_state_dict(state['model']);del state;gc.collect()
        decoder=adapter.model.decoder;attention=decoder.dec_layers[0].mha1
        traces=[]
        def hook(module,args):
            traces.append(args[0].double().diagonal(dim1=-2,dim2=-1).sum(-1).cpu().numpy())
        handle=attention.layernorm1.register_forward_pre_hook(hook)
        torch.cuda.reset_peak_memory_stats(device)
        for c in spec['cells']:
            if sha256(c['path'])!=c['sha256']:raise ValueError('Fresh source changed')
            x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))).reshape(-1,c['L'],c['L'])
            arrays={'chain':np.repeat(np.arange(32),8)}
            for depth in range(3):
                q=c['q'];L=x.shape[-1];site=L*L-1
                arrays[f'configurations-{depth}']=x
                arrays[f'fractions-{depth}']=np.stack([(x==a).mean((1,2)) for a in range(q)],1)
                stats=observables(x,q)
                arrays[f'spatial-{depth}']=np.stack([stats['energy']/L**2,stats['corr1']],1)
                meta=context_tokens(processor,q,L,1.);letters=alphabet[:q];tokens=meta+letters
                ids=torch.tensor(tokens,device=device)
                embed=decoder.embedding(ids)*torch.sqrt(torch.tensor(decoder.d_model,device=device,dtype=decoder.embedding.weight.dtype))
                query=attention.wq(decoder.layernorm1(embed)).reshape(len(tokens),heads,64)
                norms=query.double().square().sum(-1).cpu().numpy()
                arrays[f'gram-coefficients-{depth}']=norms[len(meta):].T
                arrays[f'gram-offset-{depth}']=norms[:len(meta)].sum(0)
                captured={};traces.clear()
                for j in range(0,len(x),4):
                    inputs=prefix_batch(x[j:j+4],site,meta,letters,device)
                    logits,output=selected_forward(adapter,inputs,letters,capture=True)
                    values=dict(logits=logits.cpu().numpy(),hidden=output.hidden_states[-1][:,-1].cpu().numpy(),
                        A_common=torch.stack([a[0].mean(-2) for a in output.pldr_attentions],1).cpu().numpy(),
                        G_common=torch.stack([a[5].mean(-2) for a in output.pldr_attentions],1).cpu().numpy())
                    for k,v in values.items():captured.setdefault(k,[]).append(v)
                    del output,logits;adapter.capture=False;adapter.head_outputs={}
                for k,v in captured.items():arrays[f'{k}-{depth}']=np.concatenate(v)
                arrays[f'gram-traces-{depth}']=np.concatenate(traces)
                flat=x.reshape(len(x),-1)
                arrays[f'prefix-fractions-{depth}']=np.stack([(flat[:,:site]==a).mean(1) for a in range(q)],1)
                if depth<2:x=dyadic_block(x,q,218200+10*c['id']+depth)
            np.savez_compressed(out/f'q{c["q"]}-L{c["L"]}.npz',**arrays)
            print(case['name'],c['q'],c['L'],'observed',flush=True)
        handle.remove()
        report=dict(status='complete',case=case,native_updates=0,configurations=2048,
            model_configuration_evaluations=6144,protocol_sha256=sha256(study/'protocol.json'),
            seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
            files={p.name:sha256(p) for p in out.iterdir() if p.is_file()})
        write_json(out/'manifest.json',report)
        del adapter,decoder,attention;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['profile','prepare','worker'])
    p.add_argument('--study',required=True);p.add_argument('--heads',type=int,choices=[4,8]);p.add_argument('--device')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.study)
    elif a.action=='profile':profile(a.study,a.heads,a.device)
    else:worker(a.study,a.heads,a.device)
