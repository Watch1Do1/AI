// Byte-Pair Encoding (BPE) implementation for subword tokenization

export interface BPEMerge {
  pair: [string, string];
  merged: string;
  count: number;
}

export class BPETokenizer {
  vocab: string[] = [];
  merges: BPEMerge[] = [];
  vocabMap: Map<string, number> = new Map();

  constructor(corpus?: string, numMerges: number = 50) {
    if (corpus) {
      this.train(corpus, numMerges);
    }
  }

  // Train BPE merges on a raw text corpus
  train(corpus: string, targetMerges: number = 50): void {
    this.merges = [];
    this.vocabMap.clear();

    // 1. Initial vocabulary is base characters
    const uniqueChars = Array.from(new Set(corpus.split(''))).sort();
    const vocabSet = new Set<string>(['<unk>', ...uniqueChars]);

    // Split corpus into words and represent each word as a list of characters
    // Using simple whitespace word boundary splitting
    const rawWords = corpus.split(/(\s+)/).filter(w => w.length > 0);
    let tokenizedWords: string[][] = rawWords.map(w => w.split(''));

    for (let i = 0; i < targetMerges; i++) {
      // Count frequency of all adjacent pairs across words
      const pairCounts = new Map<string, number>();

      for (const word of tokenizedWords) {
        for (let j = 0; j < word.length - 1; j++) {
          const pairKey = `${word[j]}|||${word[j + 1]}`;
          pairCounts.set(pairKey, (pairCounts.get(pairKey) || 0) + 1);
        }
      }

      if (pairCounts.size === 0) break;

      // Find the most frequent pair
      let bestPairStr = '';
      let maxCount = 0;

      for (const [pairStr, count] of pairCounts.entries()) {
        if (count > maxCount) {
          maxCount = count;
          bestPairStr = pairStr;
        }
      }

      if (maxCount < 2) break; // Stop when pairs only occur once

      const [first, second] = bestPairStr.split('|||');
      const merged = first + second;

      this.merges.push({
        pair: [first, second],
        merged,
        count: maxCount
      });
      vocabSet.add(merged);

      // Merge the pair in all words
      tokenizedWords = tokenizedWords.map(word => {
        const newWord: string[] = [];
        let j = 0;
        while (j < word.length) {
          if (j < word.length - 1 && word[j] === first && word[j + 1] === second) {
            newWord.push(merged);
            j += 2;
          } else {
            newWord.push(word[j]);
            j += 1;
          }
        }
        return newWord;
      });
    }

    this.vocab = Array.from(vocabSet);
    this.vocab.forEach((tok, idx) => {
      this.vocabMap.set(tok, idx);
    });
  }

  // Tokenize arbitrary text into subword token strings
  tokenize(text: string): string[] {
    if (this.merges.length === 0) {
      return text.split('');
    }

    const rawWords = text.split(/(\s+)/).filter(w => w.length > 0);
    let tokens: string[] = [];

    for (const rawWord of rawWords) {
      let wordTokens = rawWord.split('');

      for (const merge of this.merges) {
        const [first, second] = merge.pair;
        const newWordTokens: string[] = [];
        let i = 0;
        while (i < wordTokens.length) {
          if (i < wordTokens.length - 1 && wordTokens[i] === first && wordTokens[i + 1] === second) {
            newWordTokens.push(merge.merged);
            i += 2;
          } else {
            newWordTokens.push(wordTokens[i]);
            i += 1;
          }
        }
        wordTokens = newWordTokens;
      }

      tokens = tokens.concat(wordTokens);
    }

    return tokens;
  }

  encode(text: string): number[] {
    const subwords = this.tokenize(text);
    return subwords.map(sub => this.vocabMap.get(sub) ?? 0);
  }

  decode(tokenIds: number[]): string {
    return tokenIds.map(id => this.vocab[id] || '').join('');
  }

  getCompressionRatio(text: string): { charCount: number; tokenCount: number; ratio: number } {
    const charCount = text.length;
    const tokenCount = Math.max(1, this.encode(text).length);
    const ratio = charCount / tokenCount;
    return { charCount, tokenCount, ratio };
  }
}
