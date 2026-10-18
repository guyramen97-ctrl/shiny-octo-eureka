import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialized Gemini instance
let aiClient: GoogleGenAI | null = null;
function getAIClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "",
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

function buildSystemPrompt(botName = "Decision Helper Bot", botPersona = "Friendly & Practical Friend") {
  return `You are ${botName}, powered by the Complex Decision Engine (CDE) with Tier-2 Cognitive Intelligence. Your purpose is to help users make fast, clear, high-conviction decisions between 2 or more options using multi-dimensional cognitive analysis.

Personality & Tone
Style: Analytical, objective, practical, and conversational—like an elite strategic advisor and insightful friend (colored slightly by persona: ${botPersona}).
Rules: Zero fluff, no unsolicited preamble, no long-winded lectures. Keep sentences punchy, high-impact, and clear.

Core Instructions & Capabilities
1. Profile & Name Customization
If the user asks to change your name, personality, or profile (e.g., "Change your name to Choice Genie" or "Act like a strict coach"), acknowledge the change in 1 short sentence, adopt the new persona immediately, and apply it to all future responses.
Also, when acknowledging a name or profile change, append a hidden system tag at the very end of your response in this exact format: [BOT_UPDATE:name="Extracted Name",persona="Extracted Persona"] so the user interface can adapt.

2. Clarify First
If the user provides a vague dilemma without clear choices or context, reply only with: "What are the 2–3 options you are choosing between?"

3. Strict Information Limits (No Made-Up Facts)
Base all pros, cons, and evaluations strictly on details provided by the user and sound logical deduction. If key details are missing, state: "Based on what you shared..."

4. Complex Decision Engine - Mandatory Layout
Whenever 2 or more options/choices are provided or discussed, you MUST execute a multi-layered evaluation and use this EXACT layout:

Quick Summary: [1 sentence defining the user's decision.]

Cognitive Bias Calibration:
- Detected Bias: [Sunk Cost Fallacy | Loss Aversion | Analysis Paralysis | Confirmation Bias | Status Quo Bias | False Dilemma | None Detected]
- Manifestation: [1 sentence citing where the user's query or dilemma exhibited this cognitive pattern]
- De-Biasing Challenge: [1 sharp question forcing the user to isolate objective logic from psychological trap]

Confidence & Risk Matrix:
- Short-Term Impact: [Score 1-10]/10 - [1 brief sentence on rollout velocity and immediate impact]
- Long-Term Risk: [Score 1-10]/10 - [1 brief sentence on systemic downside or lock-in]
- Reversibility: [Score 1-10]/10 - [Two-Way Door | One-Way Door | Partial Rollback] - [1 brief sentence on unwinding difficulty]
- Confidence Score: [Percentage between 40% and 98%]
- Risk Level: [Low | Moderate | High | Critical]
- Mitigation: [1 sentence actionable risk mitigation advice]

Advisory Perspectives:
- Pragmatist: [1 sentence on immediate execution, cost-efficiency, and operational ROI]
- Visionary: [1 sentence on compounding leverage, long-term upside, and strategic options]
- Realist: [1 sentence on constraints, hidden friction, and worst-case pitfalls]

Weighted Trade-Off Matrix:
Option A: [Option Name] [Net Utility Index: +X.X]
+ Pro: [Benefit] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> +[score])
+ Pro: [Benefit] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> +[score])
- Con: [Drawback] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> -[score])
- Con: [Drawback] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> -[score])

Option B: [Option Name] [Net Utility Index: +X.X]
+ Pro: [Benefit] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> +[score])
+ Pro: [Benefit] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> +[score])
- Con: [Drawback] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> -[score])
- Con: [Drawback] (Weight: [1-5]/5, Urgency: [1.0-2.0]x -> -[score])
(If 3 choices are given, provide Option A, Option B, Option C with 1 Pro and 1 Con each)

Temporal Scenario Forecast:
- T+24 Hours (Immediate): [Immediate execution state or emotional relief]
- T+6 Months (Medium-Term): [Operational habituation, early ROI, or compounding capability]
- T+5 Years (Long-Term): [Macro trajectory, career/organizational optionality or lock-in]
- Ripple Effect: [1 non-obvious secondary consequence down the line across stakeholders]

Recommendation: [Pick One Option] (Utility Delta: +X.X) Because: [1 sentence explaining the key decisive factor.]

Devil's Advocate Challenge:
- Blind Spot: [1 overlooked factor, unproven assumption, or hidden cost]
- Counter-Argument: [The strongest sharp case AGAINST the primary recommendation]
- Stress-Test Question: [1 forcing question to stress-test conviction]

Next Step: [1 small, instant action the user can complete right now.]

Safety & Compliance Guardrail
If the decision involves medical, financial, or legal choices, append this exact line at the very end of your message: "Note: This is general advice only. For important decisions, please consult a professional."`;
}

// API Routes
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: Date.now() });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { messages, botProfile } = req.body;
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "Messages array is required." });
    }

    const ai = getAIClient();
    const botName = botProfile?.name || "Decision Helper Bot";
    const botPersona = botProfile?.persona || "Friendly & Practical Friend";
    const systemInstruction = buildSystemPrompt(botName, botPersona);

    // Format history for Gemini generateContent
    const formattedContents = messages.map((m: { role: string; content: string }) => ({
      role: m.role === "user" ? "user" : "model",
      parts: [{ text: m.content }],
    }));

    // Model fallback priority using current stable aliases
    const candidateModels = [
      "gemini-2.5-flash",
      "gemini-3.7-flash",
      "gemini-2.5-flash-lite",
      "gemini-flash-latest",
    ];
    let responseText = "";
    let lastError: any = null;

    for (const model of candidateModels) {
      let retries = 2; // Up to 2 retries per model for transient 503/429
      while (retries >= 0) {
        try {
          const response = await ai.models.generateContent({
            model: model,
            contents: formattedContents,
            config: {
              systemInstruction: systemInstruction,
              temperature: 0.2, // low temperature for consistent, strict adherence to rules & constraints
            },
          });

          responseText = response.text || "";
          lastError = null;
          break; // Success! Break out of retry loop
        } catch (err: any) {
          lastError = err;
          const status = err?.status || err?.code || (err?.message?.includes("503") ? 503 : 0);
          const isTransient =
            status === 503 ||
            status === 429 ||
            err?.message?.includes("high demand") ||
            err?.message?.includes("overloaded") ||
            err?.message?.includes("UNAVAILABLE") ||
            err?.message?.includes("ResourceExhausted");

          if (isTransient && retries > 0) {
            const delay = (3 - retries) * 1000 + Math.floor(Math.random() * 500);
            console.warn(`[Gemini API] Model ${model} returned transient error (${status || 'transient'}). Retrying in ${delay}ms... (${retries} retries left)`);
            await new Promise((resolve) => setTimeout(resolve, delay));
            retries--;
          } else {
            console.warn(`[Gemini API] Model ${model} failed, attempting next available model:`, err?.message || err);
            break; // Try next fallback candidate model
          }
        }
      }

      if (responseText) {
        break; // Successfully got response
      }
    }

    if (!responseText && lastError) {
      // Clean up error message if it is JSON
      let userFriendlyMessage = "The AI service is temporarily experiencing high demand. Please try again in a few moments.";
      try {
        if (typeof lastError?.message === "string") {
          if (lastError.message.includes("{")) {
            const jsonStart = lastError.message.indexOf("{");
            const parsed = JSON.parse(lastError.message.slice(jsonStart));
            if (parsed?.error?.message) {
              userFriendlyMessage = parsed.error.message;
            }
          } else {
            userFriendlyMessage = lastError.message;
          }
        }
      } catch {
        // Fallback message
      }

      return res.status(503).json({
        error: userFriendlyMessage,
      });
    }

    // Check if there is a bot update tag
    let updatedProfile: { name?: string; persona?: string } | null = null;
    const botUpdateMatch = responseText.match(/\[BOT_UPDATE:name="([^"]+)",persona="([^"]+)"\]/i);
    if (botUpdateMatch) {
      updatedProfile = {
        name: botUpdateMatch[1],
        persona: botUpdateMatch[2],
      };
      // Clean tag from user-facing text
      responseText = responseText.replace(/\[BOT_UPDATE:[^\]]+\]/gi, "").trim();
    } else {
      // Fallback regex in case formatting differed
      const nameMatch = responseText.match(/\[BOT_UPDATE:name="([^"]+)"\]/i);
      if (nameMatch) {
        updatedProfile = { name: nameMatch[1] };
        responseText = responseText.replace(/\[BOT_UPDATE:[^\]]+\]/gi, "").trim();
      }
    }

    return res.json({
      text: responseText,
      updatedProfile,
    });
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    return res.status(500).json({
      error: error?.message || "Failed to generate decision assistance. Please try again.",
    });
  }
});

// Start Express server and mount Vite
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
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
    console.log(`Decision Helper Bot server running on port ${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Critical server startup error:", err);
  process.exit(1);
});
