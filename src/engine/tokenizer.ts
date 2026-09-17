import { TokenizerData } from '../types';

export class CharTokenizer {
  private charToId: Map<string, number> = new Map();
  private idToChar: Map<number, string> = new Map();
  private unkTokenId: number = 0;
  private chars: string[] = [];

  constructor(corpus?: string) {
    if (corpus) {
      this.train(corpus);
    }
  }

  train(corpus: string): void {
    this.charToId.clear();
    this.idToChar.clear();

    // Unique sorted characters
    const uniqueChars = Array.from(new Set(corpus.split(''))).sort();
    
    // Reserve [UNK] if not in chars
    const specialChars = ['<unk>'];
    this.chars = [...specialChars, ...uniqueChars];

    this.chars.forEach((ch, idx) => {
      this.charToId.set(ch, idx);
      this.idToChar.set(idx, ch);
    });

    this.unkTokenId = 0;
  }

  encode(text: string): number[] {
    const tokens: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const id = this.charToId.get(ch);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        tokens.push(this.unkTokenId);
      }
    }
    return tokens;
  }

  decode(tokens: number[]): string {
    let result = '';
    for (const token of tokens) {
      if (token === this.unkTokenId) {
        result += '?';
      } else {
        result += this.idToChar.get(token) ?? '';
      }
    }
    return result;
  }

  get vocabSize(): number {
    return this.chars.length;
  }

  getCharList(): string[] {
    return [...this.chars];
  }

  getData(): TokenizerData {
    const charToIdObj: Record<string, number> = {};
    const idToCharObj: Record<number, string> = {};
    this.charToId.forEach((val, key) => { charToIdObj[key] = val; });
    this.idToChar.forEach((val, key) => { idToCharObj[key] = val; });

    return {
      charToId: charToIdObj,
      idToChar: idToCharObj,
      vocabSize: this.chars.length,
      chars: this.chars
    };
  }
}
