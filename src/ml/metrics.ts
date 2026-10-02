export function sigmoid(x: number): number {
  if (x < -35) return 0;
  if (x > 35) return 1;
  return 1 / (1 + Math.exp(-x));
}

export function logLoss(yTrue: number[], yPred: number[]): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.max(1e-15, Math.min(1 - 1e-15, yPred[i]));
    sum += yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p);
  }
  return -sum / yTrue.length;
}

export function accuracy(yTrue: number[], yPred: number[]): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++;
  }
  return correct / yTrue.length;
}

export function confusionMatrix(yTrue: number[], yPred: number[]) {
  let tn = 0, fp = 0, fn = 0, tp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else tn++;
  }
  return { tn, fp, fn, tp };
}

export function precision(yTrue: number[], yPred: number[]): number {
  const cm = confusionMatrix(yTrue, yPred);
  return cm.tp + cm.fp === 0 ? 0 : cm.tp / (cm.tp + cm.fp);
}

export function recall(yTrue: number[], yPred: number[]): number {
  const cm = confusionMatrix(yTrue, yPred);
  return cm.tp + cm.fn === 0 ? 0 : cm.tp / (cm.tp + cm.fn);
}

export function f1Score(yTrue: number[], yPred: number[]): number {
  const p = precision(yTrue, yPred);
  const r = recall(yTrue, yPred);
  return p + r === 0 ? 0 : (2 * p * r) / (p + r);
}

export function mae(yTrue: number[], yPred: number[]): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i] - yPred[i]);
  }
  return sum / yTrue.length;
}

export function brierScore(yTrue: number[], yProb: number[]): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += (yProb[i] - yTrue[i]) ** 2;
  }
  return sum / yTrue.length;
}

export function rocCurve(yTrue: number[], yScore: number[]): { fpr: number; tpr: number }[] {
  const n = yTrue.length;
  const pos = yTrue.filter((y) => y === 1).length;
  const neg = n - pos;
  if (pos === 0 || neg === 0) return [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];

  const sorted = yScore.map((s, i) => ({ score: s, label: yTrue[i] })).sort((a, b) => b.score - a.score);

  const points: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0, fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].label === 1) tp++;
    else fp++;
    const tpr = tp / pos;
    const fpr = fp / neg;
    points.push({ fpr, tpr });
  }
  return points;
}

export function rocAuc(yTrue: number[], yScore: number[]): number {
  const pos = yTrue.filter((y) => y === 1).length;
  const neg = yTrue.length - pos;
  if (pos === 0 || neg === 0) return 0.5;

  // AUC = P(score_pos > score_neg) using rank-based formula
  const indexed = yScore.map((s, i) => ({ score: s, label: yTrue[i] }));
  indexed.sort((a, b) => a.score - b.score);

  let rankSum = 0;
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length - 1 && indexed[j + 1].score === indexed[i].score) j++;
    const avgRank = (i + j) / 2 + 1; // 1-indexed average rank
    for (let k = i; k <= j; k++) {
      if (indexed[k].label === 1) rankSum += avgRank;
    }
    i = j + 1;
  }

  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

export function prCurve(yTrue: number[], yScore: number[]): { recall: number; precision: number }[] {
  const pos = yTrue.filter((y) => y === 1).length;
  if (pos === 0) return [{ recall: 0, precision: 1 }, { recall: 1, precision: 0 }];

  const sorted = yScore.map((s, i) => ({ score: s, label: yTrue[i] })).sort((a, b) => b.score - a.score);

  const points: { recall: number; precision: number }[] = [{ recall: 0, precision: 1 }];
  let tp = 0, fp = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].label === 1) tp++;
    else fp++;
    const recall = tp / pos;
    const precision = tp / (tp + fp);
    points.push({ recall, precision });
  }
  return points;
}

export function prAuc(yTrue: number[], yScore: number[]): number {
  const curve = prCurve(yTrue, yScore);
  let area = 0;
  for (let i = 1; i < curve.length; i++) {
    const w = curve[i].recall - curve[i - 1].recall;
    area += w * (curve[i].precision + curve[i - 1].precision) / 2;
  }
  return area;
}

export function calibrationBins(
  yTrue: number[],
  yProb: number[],
  nBins: number = 10
): { binStart: number; binEnd: number; meanPredicted: number; meanObserved: number; count: number }[] {
  const bins: { binStart: number; binEnd: number; meanPredicted: number; meanObserved: number; count: number }[] = [];
  for (let b = 0; b < nBins; b++) {
    const start = b / nBins;
    const end = (b + 1) / nBins;
    const indices: number[] = [];
    for (let i = 0; i < yProb.length; i++) {
      if (yProb[i] >= start && (yProb[i] < end || (b === nBins - 1 && yProb[i] <= end))) {
        indices.push(i);
      }
    }
    if (indices.length === 0) {
      bins.push({ binStart: start, binEnd: end, meanPredicted: (start + end) / 2, meanObserved: 0, count: 0 });
    } else {
      const meanPred = indices.reduce((s, i) => s + yProb[i], 0) / indices.length;
      const meanObs = indices.reduce((s, i) => s + yTrue[i], 0) / indices.length;
      bins.push({ binStart: start, binEnd: end, meanPredicted: meanPred, meanObserved: meanObs, count: indices.length });
    }
  }
  return bins;
}

export function optimalThreshold(yTrue: number[], yScore: number[]): number {
  const sorted = yScore.map((s, i) => ({ score: s, label: yTrue[i] })).sort((a, b) => b.score - a.score);
  let bestF1 = 0;
  let bestThreshold = 0.5;
  let tp = 0, fp = 0;
  const pos = sorted.filter((s) => s.label === 1).length;
  const neg = sorted.length - pos;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].label === 1) tp++;
    else fp++;
    const p = tp / (tp + fp);
    const r = tp / pos;
    const f1 = p + r === 0 ? 0 : (2 * p * r) / (p + r);
    if (f1 > bestF1) {
      bestF1 = f1;
      bestThreshold = sorted[i].score;
    }
  }
  return bestThreshold;
}
