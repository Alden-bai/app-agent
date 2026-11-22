"""Audited immutable episode copies and physical-only post-action snapshots."""
import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness_awm.io import read_json,write_json,file_hash,verify_seal,db_fingerprint


def recovery_judge():
    name='harness_awm.recovery_judge_runtime_external'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/recovery_judge_runtime.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def copy_sealed_episode(source,destination):
    source=Path(source).resolve();destination=Path(destination)
    seal=verify_seal(source)
    if destination.exists():raise ValueError('destination already exists')
    destination.mkdir(parents=True)
    for name in set(seal)|{'evidence_seal.json','episode_result.json'}:
        src=(source/name).resolve()
        if not src.is_relative_to(source):raise ValueError('evidence path escape')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target)
    if verify_seal(destination)!=seal:raise ValueError('copied seal differs')
    return {'original_episode':str(source),'original_result_sha256':file_hash(source/'episode_result.json'),
            'original_seal_sha256':file_hash(source/'evidence_seal.json')}


def physical_snapshot(source,target):
    source,target=Path(source).resolve(),Path(target).resolve()
    if source==target or target.exists():raise ValueError('snapshot destination must be new')
    target.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:
        src.backup(dst)
        if dst.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('database_integrity_failed')
        violations=dst.execute('PRAGMA foreign_key_check').fetchall()
    return {'sha256':file_hash(target),'foreign_key_violations':violations,'physical_integrity':'ok'}


def install_snapshot_fix():
    from harness_awm.environment import AWMEnvAdapter
    def audit(self,label):
        path=self.folder/'audit'/(label+'.db')
        result=physical_snapshot(self.runtime/'working.db',path)
        return {'path':str(path.relative_to(self.folder)),**result,'logical_hash':db_fingerprint(path)}
    def freeze(self):
        if not self.started:raise RuntimeError('environment_preparation_incomplete')
        path=self.folder/'final.db';result=physical_snapshot(self.runtime/'working.db',path)
        write_json(self.folder/'audit/final_snapshot_integrity.json',result)
        path.chmod(0o444)
        return result['sha256']
    AWMEnvAdapter.audit=audit;AWMEnvAdapter.freeze=freeze
