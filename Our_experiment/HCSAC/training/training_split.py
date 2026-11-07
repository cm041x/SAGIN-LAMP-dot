"""Deterministic, disjoint wind splits and paired training episode schedules."""
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CATALOG=ROOT/'Our_experiment/HCSAC/Our_experiment/HCSAC/data/wind_seed_classes_5000.json'
TEST_WINDS=(11,23,4800);TEST_SEEDS=tuple(range(42,52))
CLASSES=('Low Wind','Moderate Wind','Strong Wind')

def build_split(split_seed: int=20261104) -> dict:
    data=json.loads(CATALOG.read_text());rng=random.Random(split_seed);pools={};validation=[]
    for cls in CLASSES:
        eligible=sorted(set(data['class_to_seeds'][cls])-set(TEST_WINDS))
        selected=sorted(rng.sample(eligible,3)); pools[cls]=[x for x in eligible if x not in selected]
        validation.extend({'wind_class':cls,'wind_seed':w,'terrain_seed':900000+i+len(validation)*7,'infra_seed':900000+i+len(validation)*7} for i,w in enumerate(selected))
    return {'catalog':str(CATALOG.relative_to(ROOT)),'catalog_sha256':hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            'generation_seed':split_seed,'train_wind_seed_pools':pools,'validation_combinations':validation,
            'test_combinations':[{'wind_seed':w,'terrain_seed':s,'infra_seed':s} for w in TEST_WINDS for s in TEST_SEEDS],
            'excluded_test_wind_seeds':list(TEST_WINDS),'excluded_validation_wind_seeds':[v['wind_seed'] for v in validation]}

def episode_manifest(seed: int, episodes: int, split: dict) -> list[dict]:
    rng=random.Random(seed);rows=[]
    for ep in range(episodes):
        cls=CLASSES[ep%len(CLASSES)];w=rng.choice(split['train_wind_seed_pools'][cls]);terrain=rng.randrange(100000,800000);infra=rng.randrange(100000,800000)
        rows.append({'episode':ep+1,'wind_class':cls,'wind_seed':w,'terrain_seed':terrain,'infra_seed':infra})
    return rows
