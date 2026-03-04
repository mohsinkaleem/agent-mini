/**
 * Agent Mini — WhatsApp Web Bridge
 *
 * Bridges WhatsApp Web (via whatsapp-web.js) to the Python agent over HTTP.
 *
 * Environment variables:
 *   WEBHOOK_URL  — URL the bridge POSTs incoming messages to  (default: http://localhost:18901/webhook)
 *   PORT         — port for the outgoing-message HTTP API      (default: 18902)
 *   DATA_DIR     — session / auth data directory               (default: ~/.agent-mini/whatsapp-data)
 */

const { Client, LocalAuth } = require("whatsapp-web.js");
const express = require("express");
const qrcode = require("qrcode-terminal");
const path = require("path");
const os = require("os");
const fs = require("fs");

// ------------------------------------------------------------------
// Find system Chrome / Chromium executable
// ------------------------------------------------------------------
function findChrome() {
  const candidates = [
    // macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    // Linux
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/snap/bin/chromium",
    // Windows (for completeness)
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error(
    "Could not find Chrome or Chromium. Please install Google Chrome and try again.\n" +
    "Download: https://www.google.com/chrome/"
  );
}

const WEBHOOK_URL =
  process.env.WEBHOOK_URL || "http://localhost:18901/webhook";
const PORT = parseInt(process.env.PORT || "18902", 10);
const DATA_DIR =
  process.env.DATA_DIR ||
  path.join(os.homedir(), ".agent-mini", "whatsapp-data");

// ------------------------------------------------------------------
// WhatsApp client
// ------------------------------------------------------------------

const chromePath = findChrome();
console.log(`🌐 Using Chrome: ${chromePath}`);

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: DATA_DIR }),
  puppeteer: {
    executablePath: chromePath,
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  },
});

client.on("qr", (qr) => {
  console.log("\n📱 Scan this QR code with WhatsApp:\n");
  qrcode.generate(qr, { small: true });
  console.log("");
});

client.on("ready", () => {
  console.log("✅ WhatsApp client connected and ready!");
});

client.on("authenticated", () => {
  console.log("🔐 Authenticated successfully.");
});

client.on("auth_failure", (msg) => {
  console.error("❌ Authentication failed:", msg);
  process.exit(1);
});

// Forward incoming messages to the Python webhook
const seen = new Set(); // simple dedup

client.on("message", async (msg) => {
  if (msg.fromMe) return;

  // Deduplicate (WhatsApp can deliver the same message twice)
  const dedupKey = `${msg.from}-${msg.timestamp}-${msg.body?.slice(0, 32)}`;
  if (seen.has(dedupKey)) return;
  seen.add(dedupKey);
  // Keep the set bounded
  if (seen.size > 5000) {
    const iter = seen.values();
    for (let i = 0; i < 2500; i++) iter.next();
    const keep = new Set();
    for (const v of iter) keep.add(v);
    seen.clear();
    for (const v of keep) seen.add(v);
  }

  const from = msg.from.replace("@c.us", "").replace("@s.whatsapp.net", "");
  const body = msg.body || "";
  if (!body.trim()) return;

  console.log(`📨 ${from}: ${body.substring(0, 120)}`);

  try {
    await fetch(WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from, body, timestamp: Date.now() }),
    });
  } catch (err) {
    console.error("Webhook error:", err.message);
  }
});

client.on("disconnected", (reason) => {
  console.log("❌ Disconnected:", reason);
  process.exit(1);
});

// ------------------------------------------------------------------
// HTTP API for outgoing messages
// ------------------------------------------------------------------

const app = express();
app.use(express.json());

app.post("/send", async (req, res) => {
  const { to, message } = req.body;
  if (!to || !message) {
    return res.status(400).json({ error: 'Missing "to" or "message"' });
  }

  try {
    // Normalise chat ID
    const chatId = to.includes("@") ? to : `${to}@c.us`;
    await client.sendMessage(chatId, message);
    res.json({ ok: true });
  } catch (err) {
    console.error("Send error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get("/status", (_req, res) => {
  const info = client.info;
  res.json({
    status: info ? "connected" : "disconnected",
    phone: info?.wid?.user || null,
  });
});

// ------------------------------------------------------------------
// Start
// ------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`🌐 Bridge HTTP API listening on http://localhost:${PORT}`);
  console.log(`📡 Forwarding messages to ${WEBHOOK_URL}`);
  console.log("");
});

client.initialize();
