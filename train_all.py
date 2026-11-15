import sys, json, time, multiprocessing as mp
sys.path.insert(0, '/home/claude/ecoq')

def job(args):
    kind, seed = args
    import torch, ppo
    t = time.time()
    if kind == 'bc':
        ag, curve = ppo.train(seed=seed, steps=163_840, init='bc', rbase=True, kl_coef=0.3, lr0=5e-5, bc_steps=40_000)
    else:
        ag, curve = ppo.train(seed=100 + seed, steps=245_760, init='scratch', rbase=True, kl_coef=0.0)
    name = f'{kind}_s{seed}'
    torch.save(dict(pi=ag.pi.state_dict(), vf=ag.vf.state_dict(), norm=ag.norm.__dict__, obs_dim=ag.norm.m.shape[0]),
               f'results/models/{name}.pt')
    json.dump(dict(curve=curve, best_step=ag.best_step, best_val=ag.best_val, bc_acc=ag.bc_acc, time=time.time() - t),
              open(f'results/train_{name}.json', 'w'))
    return name, ag.best_step, ag.best_val, time.time() - t

if __name__ == '__main__':
    tasks = [('scratch', 0), ('bc', 0), ('scratch', 1), ('bc', 1), ('bc', 2)]
    with mp.Pool(2) as p:
        for r in p.imap_unordered(job, tasks):
            print(r, flush=True)
