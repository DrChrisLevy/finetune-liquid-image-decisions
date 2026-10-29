"""Offline contract checks. No real API credentials or provider calls."""
import asyncio
import base64
import copy
import io
import json
import time
import unittest
from collections import Counter
from unittest.mock import patch

import httpx
from PIL import Image
import server


class DemoTests(unittest.IsolatedAsyncioTestCase):
    def test_all_examples_are_published_test_records(self):
        published = json.loads((server.ROOT / 'data/published-test-labels.json').read_text())
        self.assertEqual(len(server.ROWS), 100)
        self.assertEqual(len({row['id'] for row in server.ROWS}), 100)
        self.assertEqual(Counter(row['split'] for row in server.ROWS),
                         {'test_task': 34, 'test_website': 33, 'test_domain': 33})
        for row in server.ROWS:
            self.assertEqual(row['label'], published['splits'][row['split']][row['id']])

    def test_same_evidence_without_labels(self):
        with patch.object(server, 'settings', return_value={}):
            for row in server.ROWS:
                changed = copy.deepcopy(row)
                changed['label'] = 'DO_NOT_SEND_THIS'
                changed['id'] = row['id']
                bodies = {p: server.payload_for(p, row) for p in ['liquid', 'liquid3b', 'openai', 'jev']}
                self.assertEqual(bodies['liquid'], bodies['liquid3b'])
                for p, body in bodies.items():
                    self.assertEqual(body, server.payload_for(p, changed))
                    self.assertNotIn('DO_NOT_SEND_THIS', json.dumps(body))
                self.assertEqual(bodies['openai']['input'][0]['content'][0]['data'], row['state'])
                self.assertEqual(bodies['jev']['questions'], bodies['liquid']['questions'])
                self.assertEqual({c['value']: c['description'] for c in bodies['openai']['questions'][0]['choices']}, row['question']['criteria'])
                for i, image in enumerate(bodies['liquid']['images']):
                    self.assertEqual(image, bodies['openai']['input'][0]['content'][i+1]['image_url'])
                    ref = bodies['jev']['state']['images'][i]
                    self.assertEqual(image.split(',')[1], bodies['jev']['assets'][ref]['data_base64'])
                    opened = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
                    self.assertTrue(all(280 <= n <= 1568 for n in opened.size))
                    self.assertEqual(list(opened.size), row['image_sizes'][i])

    def test_adjusted_geometry(self):
        source = server.ROOT / 'data/source-samples.jsonl'
        originals = [json.loads(x) for x in source.read_text().splitlines()]
        self.assertEqual(len(server.ROWS), len(originals))
        for row, original in zip(server.ROWS, originals):
            self.assertEqual(row['id'], original['id'])
            if not row['prepared']:
                self.assertEqual(row['question'], original['payload']['questions']['next_element'])
                continue
            ow, oh = original['payload']['state']['screenshot_size_pixels']
            sw, sh = row['source_size']
            scale = min(1, 1568/max(sw, sh))
            rw, rh = round(sw*scale), round(sh*scale)
            cw, ch = row['image_sizes'][0]
            for key, value in row['question']['criteria'].items():
                before = json.loads(original['payload']['questions']['next_element']['criteria'][key])
                after = json.loads(value)
                x,y,w,h=before.pop('bbox')
                box=after.pop('bbox')
                self.assertEqual(before,after)
                for actual, expected in zip(box,[x*rw/ow+(cw-rw)//2,y*rh/oh+(ch-rh)//2,w*rw/ow,h*rh/oh]):
                    self.assertAlmostEqual(actual,expected,places=3)

    async def test_parallel_providers_and_scoring(self):
        row=server.ROWS[0]
        choices=list(row['question']['criteria'])
        wrong=next(c for c in choices if c != row['label'])
        captured=[]
        async def mock(request):
            captured.append(request)
            await asyncio.sleep(.1)
            is_openai='openai.com' in request.url.host
            choice=wrong if is_openai else row['label']
            answer={'name':'next_element','type':'choice','choice':choice,'probabilities':{choice:1}}
            if is_openai: answer['probabilities']=[{'value':choice,'probability':1}]
            return httpx.Response(200,json={'model':'fixture','answers':[answer] if is_openai else {'next_element':answer}})
        with patch.object(server,'settings',return_value={'OPENAI_API_KEY':'test-only','TYPESAFE_API_KEY':'test-only'}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
                server.app.state.client=client
                start=time.perf_counter()
                results=await asyncio.gather(*(server.predict(p,server.Predict(index=0)) for p in ['liquid','liquid3b','openai','jev']))
                self.assertLess(time.perf_counter()-start,.28)
                self.assertEqual([r['correct'] for r in results],[True,True,False,True])
                self.assertTrue(all(r['latency_ms']>=100 for r in results))
        jev=next(r for r in captured if 'typesafe' in r.url.host)
        self.assertEqual(jev.headers['TypeSafe-Danger-Zone-Images'],'true')
        liquid=[r for r in captured if 'modal' in r.url.host]
        self.assertEqual({str(r.url) for r in liquid},
                         {server.LIQUID_URL+'/v1/systemone', server.LIQUID_3B_URL+'/v1/systemone'})
        for request in liquid:
            self.assertNotIn('authorization',request.headers)

    async def test_errors_and_secrets(self):
        with patch.object(server,'settings',return_value={}):
            with self.assertRaises(server.HTTPException) as ctx:
                await server.predict('openai',server.Predict(index=0))
            self.assertEqual(ctx.exception.status_code,409)
        async def mock(request):
            return httpx.Response(401,text='SECRET_SENTINEL')
        with patch.object(server,'settings',return_value={'OPENAI_API_KEY':'SECRET_SENTINEL'}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
                server.app.state.client=client
                with self.assertRaises(server.HTTPException) as ctx:
                    await server.predict('openai',server.Predict(index=0))
                self.assertNotIn('SECRET_SENTINEL',str(ctx.exception.detail))
                self.assertNotIn('SECRET_SENTINEL',json.dumps(server.provider_config()))
        with self.assertRaises(ValueError):
            server.normalize_answer('liquid',{'answers':{'next_element':{'choice':'invalid'}}},{'a':'A'})


if __name__ == '__main__':
    unittest.main()
