"""Audit raw thread-sweep samples and summarize measured CPU/GPU crossover points."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from bench_utils import summarize,write_csv,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    sweep=json.loads((args.run/'sweep.json').read_text(encoding='utf-8'))
    if not sweep.get('completed_utc'):
        raise ValueError('Incomplete sweep')
    all_summary=[]
    total_samples=0
    for threads in sweep['thread_order']:
        run=args.run/f'threads-{threads:02d}'
        meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        if not meta.get('completed_utc') or meta['environment']['cpu_threads']!=threads:
            raise ValueError('Incomplete run or thread mismatch')
        for name,digest in meta['source_sha256'].items():
            if hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Source hash mismatch: {name}')
        rows=[json.loads(x) for x in (run/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        expected={(batch,variant,trial) for batch in sweep['batches'] for variant in ('cpu_fp32','cpu_int8_dynamic','cuda_fp32','cuda_fp16') for trial in range(sweep['trials'])}
        actual={(r['batch'],r['variant'],r['trial']) for r in rows}
        if actual!=expected or len(rows)!=len(expected):
            raise ValueError('Missing, duplicate or unexpected configuration')
        for row in rows:
            samples=row['samples_ms']
            if row['status']!='ok' or row['profile']!='roundtrip' or row['cpu_threads']!=threads or row['seed']!=sweep['seed']+row['trial']:
                raise ValueError('Unexpected measurement settings')
            if len(samples)!=sweep['samples'] or not all(math.isfinite(x) and x>0 for x in samples):
                raise ValueError('Invalid/missing raw timing samples')
            if not math.isclose(statistics.median(samples),row['median_ms'],rel_tol=1e-9):
                raise ValueError('Median does not match raw samples')
            total_samples+=len(samples)
        all_summary.extend(summarize(rows,meta['group_fields'],meta['option_field'],sweep['trials']))
    write_csv(args.run/'all_summary.csv',all_summary)
    table=[]
    for threads in sorted(sweep['thread_order']):
        for batch in sweep['batches']:
            options={r['variant']:r for r in all_summary if r['cpu_threads']==threads and r['batch']==batch}
            feasible=[r for r in options.values() if r['relative_l2_error']<=.03 and r['status']=='ok']
            winner=min(feasible,key=lambda r:r['median_ms'])
            table.append(dict(threads=threads,batch=batch,**{variant:r['median_ms'] for variant,r in options.items()},
                              cpu_int8_speedup=options['cpu_fp32']['median_ms']/options['cpu_int8_dynamic']['median_ms'],
                              measured_fastest_error003=winner['variant'],
                              max_int8_relative_l2=options['cpu_int8_dynamic']['relative_l2_error']))
    write_csv(args.run/'crossover.csv',table)
    write_json(args.run/'verification.json',dict(configurations=len(all_summary),trial_records=len(all_summary)*sweep['trials'],
               raw_samples=total_samples,all_sources_match=True,complete_unique_records=True,
               all_sample_medians_match=True,analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    report=['# 线程数 × 批量：计入传输的线性层推理复测','',
        f"共 {len(all_summary)} 个配置、{len(all_summary)*sweep['trials']} 条按轮次记录、{total_samples:,} 个单次计时样本。CPU 线程数 1/4/8/16，batch 1/2/4/8/16/32/64/128，四种精度/设备方案，各 5 个随机种子，每组预热 20 次、计时 50 次。",'',
        '输入与输出都是 CPU FP32；GPU 路径计入阻塞拷贝、类型转换和输出返回。模型仍是随机权重 TinyMLP(768→3072→768)，不是问答模型或 HTTP 服务。以下为五轮中位数的中位数，单位 ms。','',
        '| CPU 线程 | Batch | CPU FP32 | CPU INT8 | GPU FP32 | GPU FP16 | 误差≤0.03 下实测最快 |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for r in table:
        report.append(f"| {r['threads']} | {r['batch']} | {r['cpu_fp32']:.4f} | {r['cpu_int8_dynamic']:.4f} | {r['cuda_fp32']:.4f} | {r['cuda_fp16']:.4f} | {r['measured_fastest_error003']} |")
    report+=['','## 复现与证据','',
        '```powershell','python run_thread_sweep.py --output results/my-thread-sweep',
        'python summarize_thread_sweep.py --run results/my-thread-sweep','```','',
        '- sweep.json 保存实验范围及随机线程执行顺序；各 threads-* 目录保留环境、全部单次计时、误差和源码原始字节快照。',
        '- all_summary.csv 同时给出每组最快/最慢轮次中位数、最坏轮次 P95、最大数值误差；不能把细小中位数差别解释为稳定胜出。',
        '- policy-error003.json 使用本仓库既有选择器，要求 5 轮齐全且相对 L2 误差≤0.03。这是随机 MLP 输出误差约束，不能解释为真实任务准确率最多下降 3%。',
        '- 同一线程设置内随机执行设备/精度顺序；线程设置仅以一个随机顺序分块运行，没有跨时间完全交叉平衡。笔记本频率、温度、桌面后台活动会影响结果。',
        '- CPU 线程也影响 GPU 路径的宿主调度/类型转换，所以 GPU 项目同样按线程数记录；不是 GPU 计算核心数的变化。',
        '- 本轮五个种子是 3010–3014；与旧实验的 2026–2028 不同，因此不把跨日期延迟相除称为性能提升。',
        '- 加载、网络、文本处理、排队与实际语言模型质量均不在这里；完整问答结论仍需参照配套应用实验。']
    (args.run/'README.md').write_bytes(('\n'.join(report)+'\n').encode())
    print(json.dumps(dict(configurations=len(all_summary),trial_records=len(all_summary)*sweep['trials'],raw_samples=total_samples)))


if __name__=='__main__':
    main()
