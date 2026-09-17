import { CorpusPreset } from '../types';

export const CORPUS_PRESETS: CorpusPreset[] = [
  {
    id: 'interview_script',
    title: 'Interview Script & Memory Dialogue',
    category: 'interview',
    description: 'Realistic technical interview transcripts and memory queries matching your experiment.',
    text: `INTERVIEWER: Welcome to the system evaluation. Could you introduce yourself and your expertise?
CANDIDATE: My name is Alex Rivera. I specialize in neural network architectures, distributed training, and memory systems.
INTERVIEWER: Excellent. What is the fundamental difference between foundation pretraining and memory programs?
CANDIDATE: Pretraining teaches a language model general statistical patterns and grammar from a text corpus by optimizing next-token cross-entropy loss. A memory program, conversely, stores explicit episodic facts outside the weights so they can be retrieved and injected dynamically into the context.
INTERVIEWER: Why can't a tiny model just memorize all facts during pretraining?
CANDIDATE: With only a few thousand or million parameters, a model suffers catastrophic forgetting and severe capacity limits. It cannot guarantee exact factual recall.
INTERVIEWER: How do we bridge the two?
CANDIDATE: We pretrain the base transformer to become fluent in reasoning and language structure. Then, we attach an external memory store like a vector database or key-value index that retrieves relevant memories at inference time.
INTERVIEWER: What happens when an interview transcript is processed?
CANDIDATE: The memory extractor parses key facts, timestamps, and candidate attributes into the memory bank. When a query arrives, semantic retrieval feeds the relevant memory into the prompt.
INTERVIEWER: Exactly. That keeps the foundation model decoupled from transient personal memories.`
  },
  {
    id: 'tiny_shakespeare',
    title: 'Tiny Shakespeare (nanoGPT classic)',
    category: 'literature',
    description: 'The standard 40,000-character dramatic benchmark used in Karpathy nanoGPT tutorials.',
    text: `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.`
  },
  {
    id: 'python_code',
    title: 'Tiny Python Logic & Algorithms',
    category: 'code',
    description: 'Clean Python algorithmic routines, recursion, and data structures.',
    text: `def self_attention(query, key, value, mask=None):
    # Compute scaled dot-product attention
    scores = (query @ key.T) / (query.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = softmax(scores, dim=-1)
    return weights @ value

def forward_pass(x, weights):
    # Token embedding + positional encoding
    tok_emb = weights['wte'][x]
    pos_emb = weights['wpe'][:x.shape[0]]
    h = tok_emb + pos_emb
    
    for block in weights['blocks']:
        h = block.forward(h)
        
    logits = h @ weights['lm_head']
    return logits

def train_step(model, optimizer, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()`
  },
  {
    id: 'philosophy',
    title: 'Philosophical Aphorisms',
    category: 'philosophy',
    description: 'Structured Stoic meditations and aphorisms with high syntactic regularity.',
    text: `You have power over your mind, not outside events. Realize this, and you will find strength.
Waste no more time arguing about what a good person should be. Be one.
The soul becomes dyed with the color of its thoughts.
When you arise in the morning think of what a privilege it is to be alive: to think, to enjoy, to love.
Accept the things to which fate binds you, and love the people with whom fate brings you together, but do so with all your heart.
The best revenge is to be unlike him who performed the injury.
Do not act as if you were going to live ten thousand years. Death hangs over you. While you live, while it is in your power, be good.`
  }
];
