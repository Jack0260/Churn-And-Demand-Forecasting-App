export interface TreeNode {
  isLeaf: boolean;
  prediction?: number;
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  samples?: number;
}

export interface DecisionTreeOptions {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  maxFeatures?: number;
  task?: 'classification' | 'regression';
}

export class DecisionTree {
  root: TreeNode;
  nFeatures: number;
  options: Required<DecisionTreeOptions>;
  rng: () => number;

  constructor(options: DecisionTreeOptions = {}, rng: () => number = Math.random) {
    this.options = {
      maxDepth: options.maxDepth ?? 6,
      minSamplesSplit: options.minSamplesSplit ?? 10,
      minSamplesLeaf: options.minSamplesLeaf ?? 5,
      maxFeatures: options.maxFeatures ?? -1,
      task: options.task ?? 'classification',
    };
    this.rng = rng;
    this.nFeatures = 0;
    this.root = { isLeaf: true, prediction: 0 };
  }

  fit(X: number[][], y: number[]): void {
    this.nFeatures = X[0]?.length ?? 0;
    if (this.options.maxFeatures < 0 || this.options.maxFeatures > this.nFeatures) {
      this.options.maxFeatures = this.nFeatures;
    }
    this.root = this.buildTree(X, y, 0);
  }

  private buildTree(X: number[][], y: number[], depth: number): TreeNode {
    const n = y.length;
    const prediction = this.options.task === 'classification'
      ? y.reduce((a, b) => a + b, 0) / n
      : y.reduce((a, b) => a + b, 0) / n;

    if (depth >= this.options.maxDepth || n < this.options.minSamplesSplit || n < 2) {
      return { isLeaf: true, prediction, samples: n };
    }

    const split = this.findBestSplit(X, y);
    if (!split) {
      return { isLeaf: true, prediction, samples: n };
    }

    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (X[i][split.feature] <= split.threshold) leftIdx.push(i);
      else rightIdx.push(i);
    }

    if (leftIdx.length < this.options.minSamplesLeaf || rightIdx.length < this.options.minSamplesLeaf) {
      return { isLeaf: true, prediction, samples: n };
    }

    const leftX = leftIdx.map((i) => X[i]);
    const leftY = leftIdx.map((i) => y[i]);
    const rightX = rightIdx.map((i) => X[i]);
    const rightY = rightIdx.map((i) => y[i]);

    return {
      isLeaf: false,
      feature: split.feature,
      threshold: split.threshold,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
      samples: n,
    };
  }

  private findBestSplit(X: number[][], y: number[]): { feature: number; threshold: number; gain: number } | null {
    const n = y.length;
    const parentImpurity = this.gini(y);
    let bestGain = 0;
    let bestSplit: { feature: number; threshold: number; gain: number } | null = null;

    const featureIndices = this.getFeatureSubset();

    for (const feat of featureIndices) {
      const values = X.map((row) => row[feat]).sort((a, b) => a - b);
      const unique = [...new Set(values)];

      for (let t = 0; t < unique.length - 1; t++) {
        const threshold = (unique[t] + unique[t + 1]) / 2;

        let leftCount = 0, rightCount = 0;
        let leftSum = 0, rightSum = 0;
        for (let i = 0; i < n; i++) {
          if (X[i][feat] <= threshold) {
            leftCount++;
            leftSum += y[i];
          } else {
            rightCount++;
            rightSum += y[i];
          }
        }

        if (leftCount < this.options.minSamplesLeaf || rightCount < this.options.minSamplesLeaf) continue;

        const leftY = y.filter((_, i) => X[i][feat] <= threshold);
        const rightY = y.filter((_, i) => X[i][feat] > threshold);

        const leftImp = this.gini(leftY);
        const rightImp = this.gini(rightY);
        const gain = parentImpurity - (leftCount / n) * leftImp - (rightCount / n) * rightImp;

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature: feat, threshold, gain };
        }
      }
    }

    return bestSplit;
  }

  private getFeatureSubset(): number[] {
    if (this.options.maxFeatures >= this.nFeatures) {
      return Array.from({ length: this.nFeatures }, (_, i) => i);
    }
    const indices = Array.from({ length: this.nFeatures }, (_, i) => i);
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices.slice(0, this.options.maxFeatures);
  }

  private gini(y: number[]): number {
    const n = y.length;
    if (n === 0) return 0;
    const p = y.reduce((a, b) => a + b, 0) / n;
    return 1 - p * p - (1 - p) * (1 - p);
  }

  predictProbaOne(x: number[]): number {
    let node = this.root;
    while (!node.isLeaf && node.feature !== undefined && node.threshold !== undefined) {
      if (x[node.feature] <= node.threshold) {
        node = node.left!;
      } else {
        node = node.right!;
      }
    }
    return node.prediction ?? 0;
  }

  predictProba(X: number[][]): number[] {
    return X.map((x) => this.predictProbaOne(x));
  }

  predict(X: number[][]): number[] {
    return this.predictProba(X).map((p) => (p >= 0.5 ? 1 : 0));
  }
}
