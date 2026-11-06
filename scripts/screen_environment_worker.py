#!/usr/bin/env python3
"""Network-isolated ASGI probes; one child process and fresh DB per operation."""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import sys
import time
import traceback
import types
from urllib.parse import quote as urlquote


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,default=str)+'\n')


def q(name):
    return '"'+name.replace('"','""')+'"'


def db_state(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables={}
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            values=db.execute(f'SELECT * FROM {q(name)}').fetchall()
            encoded=sorted(json.dumps(row,sort_keys=True,default=str) for row in values)
            tables[name]=dict(count=len(values),hash=hashlib.sha256('\n'.join(encoded).encode()).hexdigest())
        return dict(tables=tables,integrity=db.execute('PRAGMA integrity_check').fetchall(),
                    foreign_keys=db.execute('PRAGMA foreign_key_check').fetchall())


def load_server(work, original_db, destination, name):
    code=(work/'server.py').read_text()
    if original_db not in code:
        raise ValueError('generated server does not reference expected database; refusing probe')
    code=code.replace(original_db,str(destination))
    os.environ['DATABASE_PATH']='sqlite:///'+str(destination)
    module=types.ModuleType(name);module.__file__=str(work/'server.py')
    sys.modules[name]=module
    exec(compile(code,module.__file__,'exec'),module.__dict__)
    return module


class MissingValue(Exception):
    pass


def resolve_schema(schema,spec,seen=()):
    if '$ref' in schema:
        ref=schema['$ref']
        if not ref.startswith('#/') or ref in seen:
            raise MissingValue('external or cyclic schema reference')
        value=spec
        for part in ref[2:].split('/'):
            value=value[part.replace('~1','/').replace('~0','~')]
        return resolve_schema(value,spec,seen+(ref,))
    if 'allOf' in schema:
        merged=dict(schema);merged.pop('allOf')
        for part in schema['allOf']:
            child=resolve_schema(part,spec,seen)
            merged['properties']={**merged.get('properties',{}),**child.get('properties',{})}
            merged['required']=list(dict.fromkeys(merged.get('required',[])+child.get('required',[])))
            merged.update({k:v for k,v in child.items() if k not in ('properties','required')})
        return merged
    for union in ('anyOf','oneOf'):
        if union in schema:
            choices=[x for x in schema[union] if x.get('type')!='null']
            if choices:
                return {**{k:v for k,v in schema.items() if k!=union},**resolve_schema(choices[0],spec,seen)}
    return schema


def seed_binding(db,name,description):
    tables={r[0]:{x[1] for x in db.execute(f'PRAGMA table_xinfo({q(r[0])})')}
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    pairs=[]
    for table,column in re.findall(r'\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b',description):
        if table in tables and column in tables[table]:pairs.append((table,column))
    if pairs:
        pairs=list(dict.fromkeys(pairs))
        return pairs[0] if len(pairs)==1 else None
    if name.endswith('_id'):
        base=name[:-3]
        for table in (base,base+'s',base+'es',base[:-1]+'ies' if base.endswith('y') else ''):
            if table in tables and 'id' in tables[table]:pairs.append((table,'id'))
    if pairs:
        pairs=list(dict.fromkeys(pairs))
        return pairs[0] if len(pairs)==1 else None
    for table,columns in tables.items():
        if name in columns and name!='id':pairs.append((table,name))
    pairs=list(dict.fromkeys(pairs))
    return pairs[0] if len(pairs)==1 else None


def seed_value(db,name,description):
    binding=seed_binding(db,name,description)
    if binding:
        table,column=binding
        row=db.execute(f'SELECT {q(column)} FROM {q(table)} WHERE {q(column)} IS NOT NULL ORDER BY {q(column)} LIMIT 1').fetchone()
        if row:return row[0],f'seed:{table}.{column}'
    raise MissingValue('no unambiguous seeded value for '+name)


def bind_references(value,schema,spec,db,name,provenance):
    """Examples are documentation, not evidence that their entity IDs exist."""
    schema=resolve_schema(schema,spec)
    if isinstance(value,dict):
        return {k:bind_references(v,schema.get('properties',{}).get(k,{}),spec,db,k,provenance) for k,v in value.items()}
    if isinstance(value,list):
        return [bind_references(v,schema.get('items',{}),spec,db,name,provenance) for v in value]
    if value is not None and name.endswith(('_id','_code')):
        binding=seed_binding(db,name,schema.get('description',''))
        if binding:
            table,column=binding
            exists=db.execute(f'SELECT 1 FROM {q(table)} WHERE {q(column)}=? LIMIT 1',(value,)).fetchone()
            if exists:
                provenance.append(dict(field=name,source='verified_seed_reference',table=table,column=column,value=value))
            else:
                try:
                    replacement,source=seed_value(db,name,schema.get('description',''))
                    provenance.append(dict(field=name,source=source,example_replaced=value,value=replacement))
                    return replacement
                except MissingValue:
                    provenance.append(dict(field=name,source='unresolved_empty_seed_reference',value=value))
    return value


def make_value(schema,spec,db,name,provenance,depth=0,hint=None):
    if depth>8:raise MissingValue('schema nesting limit')
    schema=resolve_schema(schema,spec)
    for key in ('example','default'):
        if key in schema and schema[key] is not None:
            provenance.append(dict(field=name,source='schema_'+key))
            return schema[key]
    if schema.get('examples'):
        provenance.append(dict(field=name,source='schema_example'));return schema['examples'][0]
    if hint is not None:
        provenance.append(dict(field=name,source='public_contract_example'));return hint
    if 'const' in schema:return schema['const']
    if schema.get('enum'):
        provenance.append(dict(field=name,source='schema_enum'));return schema['enum'][0]
    typ=schema.get('type')
    if typ=='object' or 'properties' in schema:
        result={}
        for prop in schema.get('required',[]):
            result[prop]=make_value(schema.get('properties',{}).get(prop,{}),spec,db,prop,provenance,depth+1)
        return result
    if typ=='array':
        size=max(1,schema.get('minItems',0))
        if size>10:raise MissingValue('large required array')
        return [make_value(schema['items'],spec,db,name,provenance,depth+1) for _ in range(size)]
    value,source=seed_value(db,name,schema.get('description',''))
    if typ=='boolean' and value in (0,1):value=bool(value)
    provenance.append(dict(field=name,source=source))
    return value


def build_cases(spec,contract,initial):
    from jsonschema import Draft202012Validator, RefResolver
    hints={op['operation_id']:op for group in contract['api_spec']['api_groups'] for op in group['endpoints']}
    cases,skipped=[],[]
    with sqlite3.connect(initial.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for path,methods in spec['paths'].items():
            for method,op in methods.items():
                if method not in ('get','post','put','patch','delete'):continue
                tool=op.get('operationId',method+' '+path)
                case=dict(tool=tool,method=method.upper(),path_template=path,path=path,params={},provenance=[])
                required=hints.get(tool,{}).get('request_params',{})
                try:
                    params=methods.get('parameters',[])+op.get('parameters',[])
                    for param in params:
                        param=resolve_schema(param,spec)
                        if not param.get('required') and param['in']!='path':continue
                        name=param['name'];schema={**param.get('schema',{})}
                        if 'description' in param:schema.setdefault('description',param['description'])
                        hint=param.get('example',required.get(name,{}).get('example'))
                        value=make_value(schema,spec,db,name,case['provenance'],hint=hint)
                        value=bind_references(value,schema,spec,db,name,case['provenance'])
                        Draft202012Validator(schema,resolver=RefResolver.from_schema(spec)).validate(value)
                        if param['in']=='path':case['path']=case['path'].replace('{'+name+'}',urlquote(str(value),safe=''))
                        elif param['in']=='query':case['params'][name]=value
                        else:raise MissingValue('unsupported parameter location:'+param['in'])
                    body=op.get('requestBody')
                    if body:
                        body=resolve_schema(body,spec)
                        if 'application/json' not in body.get('content',{}):raise MissingValue('non-JSON request body')
                        schema=body['content']['application/json'].get('schema',{})
                        # Recursively add public-contract body examples where the schema omitted them.
                        schema=json.loads(json.dumps(resolve_schema(schema,spec)))
                        for name,child in schema.get('properties',{}).items():
                            if 'example' not in child and name in required and 'example' in required[name]:
                                child['example']=required[name]['example']
                        case['json']=make_value(schema,spec,db,'body',case['provenance'],hint=body['content']['application/json'].get('example'))
                        case['json']=bind_references(case['json'],schema,spec,db,'body',case['provenance'])
                        Draft202012Validator(schema,resolver=RefResolver.from_schema(spec)).validate(case['json'])
                    if '{' in case['path']:raise MissingValue('unresolved path parameter')
                    cases.append(case)
                except Exception as exc:
                    skipped.append(dict(tool=tool,method=method.upper(),path=path,reason=f'{type(exc).__name__}: {str(exc)[:500]}'))
    return cases,skipped


async def invoke(module,case):
    import httpx
    from starlette.routing import Match
    from urllib.parse import unquote
    scope=dict(type='http',method=case['method'],path=unquote(case['path']),root_path='')
    actual=next((r for r in module.app.routes if r.matches(scope)[0]==Match.FULL),None)
    routed=getattr(actual,'operation_id',None) or getattr(actual,'name',None)
    result=dict(actual_operation=routed,route_mismatch=routed!=case['tool'])
    async with module.app.router.lifespan_context(module.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app),base_url='http://screen.local') as client:
            response=await client.request(case['method'],case['path'],params=case['params'],**({'json':case['json']} if 'json' in case else {}))
            result.update(status=response.status_code,response_body=response.text[:64000],
                          response_bytes=len(response.content),response_sha256=hashlib.sha256(response.content).hexdigest())
    return result


def child_case(work,job,case,dest):
    result=dict(tool=case['tool'],method=case['method'],request=case)
    db=dest/'working.db';shutil.copyfile(work/'initial.db',db)
    try:
        before=db_state(db)
        module=load_server(work,job['original_db'],db,'probe_case')
        result['after_import']=db_state(db)
        try:
            result.update(asyncio.run(invoke(module,case)))
        except Exception as exc:
            result.update(exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),status=500)
        after=db_state(db)
        result['before']=before;result['after']=after
        result['changed_tables']=[t for t in set(before['tables'])|set(after['tables']) if before['tables'].get(t)!=after['tables'].get(t)]
        result['new_integrity_errors']=after['integrity']!=[('ok',)] or bool(set(map(tuple,after['foreign_keys']))-set(map(tuple,before['foreign_keys'])))
    except Exception as exc:
        result.update(exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    write(dest/'result.json',result)


def main():
    job=json.loads(Path(sys.argv[1]).read_text());work=Path(job['work'])
    summary=dict(scenario=job['scenario'],probes=[],skipped=[],counts={})
    try:
        discovery=work/'discovery.db';shutil.copyfile(work/'initial.db',discovery)
        module=load_server(work,job['original_db'],discovery,'probe_discovery')
        spec=module.app.openapi();write(work/'openapi.json',spec)
        cases,summary['skipped']=build_cases(spec,json.loads((work/'contract.json').read_text()),work/'initial.db')
        write(work/'cases.json',cases)
        for value in list(module.__dict__.values()):
            if type(value).__name__=='Engine':value.dispose()
        for i,case in enumerate(cases):
            dest=work/'calls'/f'{i:04d}';dest.mkdir(parents=True)
            pid=os.fork()
            if pid==0:
                try:
                    fd=os.open(dest/'worker.log',os.O_CREAT|os.O_WRONLY,0o600)
                    os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
                    child_case(work,job,case,dest)
                    os._exit(0)
                except BaseException:
                    traceback.print_exc();os._exit(1)
            deadline=time.monotonic()+job['timeout']
            timed_out=False
            while os.waitpid(pid,os.WNOHANG)[0]==0:
                if time.monotonic()>deadline:
                    os.kill(pid,signal.SIGKILL);os.waitpid(pid,0);timed_out=True;break
                time.sleep(.02)
            result_path=dest/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else dict(tool=case['tool'],method=case['method'],timeout=timed_out,exception=None if timed_out else 'probe_child_died')
            if not result_path.exists():write(result_path,result)
            summary['probes'].append({k:result[k] for k in ['tool','method','status','exception','timeout','route_mismatch','new_integrity_errors','changed_tables'] if k in result}|{'result_ref':str(result_path)})
        probes=summary['probes']
        summary['counts']=dict(operations=len(probes)+len(summary['skipped']),probed=len(probes),
            skipped=len(summary['skipped']),reads=sum(p['method']=='GET' for p in probes),
            writes=sum(p['method']!='GET' for p in probes),http_2xx=sum(200<=p.get('status',0)<300 for p in probes),
            http_4xx=sum(400<=p.get('status',0)<500 for p in probes),
            http_5xx=sum(p.get('status',0)>=500 for p in probes),
            exceptions=sum(bool(p.get('exception')) for p in probes),timeouts=sum(bool(p.get('timeout')) for p in probes),
            route_mismatches=sum(bool(p.get('route_mismatch')) for p in probes),
            integrity_errors=sum(bool(p.get('new_integrity_errors')) for p in probes))
    except Exception as exc:
        summary['setup_error']=f'{type(exc).__name__}: {exc}';summary['traceback']=traceback.format_exc()
    write(work/'result.json',summary)


if __name__=='__main__':main()
