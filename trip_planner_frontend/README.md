# Trip Planner Frontend

A lightweight React + Vite workspace that mirrors conversational trip-planning tools like [Mindtrip](https://mindtrip.ai/chat/) and [Airial](https://www.airial.travel/chat). The UI pairs a chat-inspired request lane with rich itinerary cards, booking links, and source attribution so travellers can move from inspiration to checkout in one place.

## Getting started

```bash
cd trip_planner_frontend
npm install
npm run dev
```

The development server defaults to `http://localhost:5173`.

### Connecting to the orchestrator API

Set the backend endpoint that exposes `POST /api/plan` using a Vite environment variable:

```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8000
```

When the value is missing or the request fails, the app falls back to an interactive sample itinerary so the interface stays explorable offline.

## Project structure

```
trip_planner_frontend/
├── index.html             # Single-page app shell
├── package.json           # React + Vite dependencies and scripts
├── src/
│   ├── App.jsx            # High-level layout and data fetching
│   ├── components/        # Chat, bundles, booking links, and source widgets
│   ├── lib/sampleData.js  # Sample TripResponse mirroring the orchestrator schema
│   └── index.css          # Global typography and resets
└── vite.config.js         # Vite dev/build configuration
```

## Design notes

- **Dual-column canvas** – The left column captures the conversation and traveller inputs. The right column visualises the generated trip with booking shortcuts, bundle comparisons, and attribution.
- **Booking-first UX** – Every bundle exposes structured booking links (stays, rail, flights, passes, experiences) so users can jump straight into checkout flows.
- **Source transparency** – Web sources and language-model backfills are displayed alongside the itinerary, mirroring the backend contract that tracks citation metadata.

## Production build

```bash
npm run build
npm run preview
```

The Vite build emits static assets in `dist/` which can be hosted on any static file service or wired into your Python application as an embedded frontend.
