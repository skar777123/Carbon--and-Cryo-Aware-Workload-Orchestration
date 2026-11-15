import sys, json, time; sys.path.insert(0,'/home/claude/ecoq')
import numpy as np, torch, env, policies, ppo
t=time.time()
ag,curve = ppo.train(seed=11, steps=307_200, verbose=True, bc_steps=40000)
torch.save(dict(pi=ag.pi.state_dict(), vf=ag.vf.state_dict(), norm=ag.norm.__dict__), 'results/models/pilot.pt')
json.dump(curve, open('results/pilot_curve.json','w'))
E = env.Env(); pp = ppo.agent_policy(ag)
out=[]
for th in (0.35,0.5):
  for lam in (2.0,4.0,6.0):
    for name,pol in (('Planner',policies.planner),('PPO',pp)):
        S=[policies.run_episode(E,pol,9000+i,theta=th,regime=rg,lam_per_h=lam) for i in range(3) for rg in ('solar','wind','fossil')]
        m=lambda k:np.mean([s[k] for s in S])
        print(f"th={th} lam={lam} {name:8s} Edyn={m('E_dyn'):8.1f} Cdyn={m('C_dyn'):7.1f} miss={m('slo_miss'):.3f} cost={m('cost'):.1f}", flush=True)
print('total', time.time()-t, 'best', ag.best_step, ag.best_val)
