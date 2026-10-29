import test from 'node:test';
import assert from 'node:assert/strict';
import {newEvaluation,recordResult,evaluationStats} from './src/evaluation.js';

test('independent provider totals update with each returned answer',()=>{
  let run=newEvaluation('a',['liquid','liquid3b','openai','jev'],100);
  for(const result of [
    {provider:'liquid',index:1,correct:true,latency_ms:200},
    {provider:'liquid3b',index:1,correct:false,latency_ms:250},
    {provider:'openai',index:1,correct:false,latency_ms:400},
    {provider:'liquid',index:0,correct:false,latency_ms:300},
    {provider:'jev',index:1,error:'Timeout'},
  ])run=recordResult(run,'a',result);
  assert.deepEqual(evaluationStats(run,'liquid'),{correct:1,answered:2,errors:0,accuracy:50,median:250});
  assert.deepEqual(evaluationStats(run,'liquid3b'),{correct:0,answered:1,errors:0,accuracy:0,median:250});
  assert.equal(evaluationStats(run,'openai').accuracy,0);
  assert.deepEqual(evaluationStats(run,'jev'),{correct:0,answered:0,errors:1,accuracy:null,median:null});
});
test('duplicate deliveries cannot inflate accuracy or sample count',()=>{
  let run=newEvaluation('a',['liquid'],100);
  const result={provider:'liquid',index:0,correct:true,latency_ms:200};
  run=recordResult(recordResult(run,'a',result),'a',result);
  assert.equal(evaluationStats(run,'liquid').answered,1);
});
test('new runs are empty and ignore late results from older runs',()=>{
  const fresh=newEvaluation('new',['liquid','openai','jev'],100);
  assert.equal(recordResult(fresh,'old',{provider:'liquid',index:0,correct:true}),fresh);
  assert.equal(evaluationStats(fresh,'liquid').accuracy,null);
  assert.equal(evaluationStats(null,'liquid').answered,0);
});
