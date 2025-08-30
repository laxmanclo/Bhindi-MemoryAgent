import express from 'express';
import { config } from 'dotenv';
import { AppController } from './controllers/appController.js';
import swaggerUi from 'swagger-ui-express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Load environment variables
config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create Express app
const app = express();
const port = process.env.PORT || 3000;
const appController = new AppController();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS handling
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Load tools configuration
const toolsConfigPath = path.join(__dirname, 'config', 'tools.json');
let toolsConfig;
try {
  const toolsConfigRaw = fs.readFileSync(toolsConfigPath, 'utf8');
  toolsConfig = JSON.parse(toolsConfigRaw);
  console.log(`Loaded tools configuration from ${toolsConfigPath}`);
} catch (error) {
  console.error(`Error loading tools configuration: ${error}`);
  toolsConfig = { tools: [] };
}

// Endpoint to get tools
app.get('/tools', (req, res) => {
  res.json(toolsConfig);
});

// Endpoint to execute tools
app.post('/tools/:toolName', async (req, res) => {
  await appController.handleTool(req, res);
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    environment: process.env.NODE_ENV,
    timestamp: new Date().toISOString()
  });
});

// Agent resource endpoint
app.post('/resource', (req, res) => {
  res.json({
    success: true,
    responseType: 'text',
    data: {
      text: 'Bhindi Agent Memory Service',
      description: 'This agent provides tools for working with the MemoryOS system.',
      version: '1.0.0',
      memoryosApiUrl: process.env.MEMORYOS_API_URL || 'http://localhost:5000'
    }
  });
});

// Start server
app.listen(port, () => {
  console.log(`Bhindi agent running on http://localhost:${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`MemoryOS API URL: ${process.env.MEMORYOS_API_URL || 'http://localhost:5000'}`);
});
