import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Lazy initialization of Gemini client
let aiClient: GoogleGenAI | null = null;
function getAI(): GoogleGenAI | null {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "5mb" }));

  // API status check
  app.get("/api/health", (req, res) => {
    res.json({
      status: "ok",
      hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
      timestamp: new Date().toISOString()
    });
  });

  // Synthesize domain corpus (e.g. interview scripts or specialized dialogues)
  app.post("/api/synthesize-corpus", async (req, res) => {
    try {
      const { topic = "technical interview transcripts", length = "medium" } = req.body;
      const ai = getAI();

      if (!ai) {
        // Fallback realistic synthesis when no key is provided
        return res.json({
          source: "fallback",
          corpus: generateFallbackCorpus(topic)
        });
      }

      const prompt = `Generate a realistic, consistent text dataset for pre-training a tiny toy language model from scratch.
Topic/Domain: "${topic}".
Format: Raw plain text without markdown headers, bullet points, or commentary.
Length: Approximately ${length === "short" ? "300 words" : length === "long" ? "1200 words" : "600 words"}.
Ensure repetitive structural patterns (e.g., Q&A, character lines, or code structures) so a tiny GPT can rapidly discover syntactic patterns and lower its loss quickly.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: prompt
      });

      const text = response.text || "";
      res.json({
        source: "gemini",
        corpus: text.trim()
      });
    } catch (err: any) {
      console.error("Failed to synthesize corpus with Gemini:", err);
      // Fallback gracefully
      res.json({
        source: "fallback-error",
        corpus: generateFallbackCorpus(req.body?.topic || "dialogue")
      });
    }
  });

  // Explain transformer mechanics & memory connection
  app.post("/api/ask-concept", async (req, res) => {
    try {
      const { question } = req.body;
      const ai = getAI();

      if (!ai) {
        return res.json({
          answer: getLocalExplanation(question)
        });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash",
        contents: `You are an expert deep learning engineer specializing in training transformers from scratch (nanoGPT, Andrej Karpathy's minGPT, PyTorch).
Answer concisely (2-3 short paragraphs max) with clear intuition:
Question: ${question}`
      });

      res.json({
        answer: response.text || "No response generated."
      });
    } catch (err: any) {
      console.error("AI question failed:", err);
      res.json({
        answer: getLocalExplanation(req.body?.question || "")
      });
    }
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
  });
}

function generateFallbackCorpus(topic: string): string {
  return `INTERVIEWER: Welcome to the evaluation session. Could you describe your background?
CANDIDATE: I have spent five years working on distributed systems and neural networks.
INTERVIEWER: What happens during the forward pass of a transformer block?
CANDIDATE: Input tokens are converted to embeddings with positional encodings added.
INTERVIEWER: And then?
CANDIDATE: The representations pass through multi-head causal self-attention, layer normalization, and a position-wise feed-forward network.
INTERVIEWER: Why is the causal mask necessary in autoregressive generation?
CANDIDATE: Because tokens at step t should never attend to future tokens at step t+1.
INTERVIEWER: How do you attach an external memory program to a frozen foundation model?
CANDIDATE: You store key facts in a vector store or episodic key-value cache, then inject retrieved memories into the prompt context.
INTERVIEWER: Exactly. Pretrained weights store statistical distributions; memory programs store explicit facts.
CANDIDATE: That is why you decouple foundation pretraining from episodic memory retrieval.`;
}

function getLocalExplanation(query: string): string {
  const q = query.toLowerCase();
  if (q.includes("loss") || q.includes("cross entropy")) {
    return "Initial cross-entropy loss for a random model is roughly -ln(1 / vocab_size). For a character vocabulary of 64 tokens, -ln(1/64) ≈ 4.15. As the transformer learns letter frequencies and bigrams, loss quickly drops to ~2.5, then ~1.8 as words and syntax form.";
  }
  if (q.includes("memory") || q.includes("interview")) {
    return "A tiny language model pretrained from scratch only learns statistical patterns (syntax, common word sequences). It does not have the capacity to reliably store long-term factual memories without massive scale or catastrophic forgetting. Decoupling your 'memory program' (like RAG, episodic KV stores, or context-injection) allows the model to act as a linguistic processor while the memory program supplies factual grounding.";
  }
  return "In a decoder-only GPT, tokens are embedded, augmented with positional vectors, and passed through masked multi-head attention and feed-forward MLP layers. Next-token prediction optimizes cross-entropy loss via backpropagation.";
}

startServer();
