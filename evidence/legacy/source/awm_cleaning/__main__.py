from __future__ import annotations
import argparse,asyncio,json
from pathlib import Path
from .common import REVISION

def main():
 parser=argparse.ArgumentParser(description='Frozen official AWM SQL+LLM cleaning; no pure-code dependency')
 parser.add_argument('--run-dir',default=f'cleaning_runs/{REVISION}')
 sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('prepare');p.add_argument('--source');p.add_argument('--revision',default=REVISION)
 r=sub.add_parser('run');r.add_argument('--scope',choices=('pilot100','all'),default='pilot100');r.add_argument('--resume',action='store_true');r.add_argument('--max-api-requests',type=int);r.add_argument('--authorize-all-model-budget',action='store_true',help='Separate explicit budget authorization; scope alone is not authorization')
 sub.add_parser('report');sub.add_parser('publish')
 args=parser.parse_args()
 if args.command=='prepare':
  from .assets import prepare
  m=prepare(args.run_dir,args.source,args.revision);result={'tasks':m['frame_tasks'],'environments':m['frame_environments'],'pilot100':len(m['pilot100']),'freeze_sha256':m['freeze_sha256']}
 elif args.command=='run':
  from .runner import run
  result=asyncio.run(run(args.run_dir,args.scope,args.resume,args.max_api_requests,args.authorize_all_model_budget))
 elif args.command=='report':
  from .report import rebuild
  result=rebuild(args.run_dir)
 else:
  from .report import publish
  result=publish(args.run_dir)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
