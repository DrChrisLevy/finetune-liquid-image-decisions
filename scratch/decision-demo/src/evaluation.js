export function newEvaluation(id, providers, total) {
  return {id, providers, total, completed: 0, status: 'running', results: {}};
}

export function recordResult(run, id, result) {
  if (!run || run.id !== id || !run.providers.includes(result.provider)) return run;
  const key = `${result.provider}:${result.index}`;
  return {...run, results: {...run.results, [key]: result}};
}

export function evaluationStats(run, provider) {
  const records = Object.values(run?.results || {}).filter(r => r.provider === provider);
  const answered = records.filter(r => !r.error && typeof r.correct === 'boolean');
  const correct = answered.filter(r => r.correct).length;
  const times = answered.map(r => r.latency_ms).filter(Number.isFinite).sort((a,b) => a-b);
  return {
    correct, answered: answered.length, errors: records.filter(r => r.error).length,
    accuracy: answered.length ? 100 * correct / answered.length : null,
    median: times.length ? (times[Math.floor((times.length-1)/2)] + times[Math.floor(times.length/2)]) / 2 : null,
  };
}
