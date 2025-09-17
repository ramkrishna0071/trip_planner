export const sampleRequest = {
  origin: 'San Francisco',
  destinations: ['Paris', 'Amsterdam', 'Berlin'],
  dates: {
    start: '2025-10-10',
    end: '2025-10-20',
  },
  budget_total: 4500,
  currency: 'USD',
  party: {
    adults: 2,
    children: 1,
    seniors: 0,
  },
  prefs: {
    objective: 'balanced',
    flexible_days: 0,
    max_flight_hours: 12,
    diet: ['vegetarian'],
    mobility: 'normal',
  },
};

export const sampleResponse = {
  trip_plan: {
    title: 'Autumn escape: Paris, Amsterdam & Berlin',
    overview:
      'Ten autumn nights weaving Parisian art, Dutch canals, and Berlin history with vegetarian-friendly picks for the whole family.',
    summary_points: [
      'Fly into Paris for four nights, then hop north by high-speed rail.',
      'Blend headline museums with family-flexible afternoons and market time.',
      'Wrap in Berlin with hands-on history walks and bikeable neighborhoods.',
    ],
    booking_links: [
      {
        category: 'flight',
        label: 'SFO → CDG nonstop (Air France)',
        url: 'https://www.airfrance.us/',
        details: 'Sample fare for 10 Oct 2025 departure, return 20 Oct 2025.',
      },
      {
        category: 'train',
        label: 'Paris → Amsterdam (Eurostar, 14 Oct)',
        url: 'https://www.eurostar.com/us-en',
        details: 'Select 09:25 departure to arrive by lunch at Amsterdam Centraal.',
      },
      {
        category: 'train',
        label: 'Amsterdam → Berlin (ICE, 17 Oct)',
        url: 'https://www.bahn.com/en/offers/regional/ic',
        details: 'Direct ICE 1592 typically from €39 per adult when booked early.',
      },
      {
        category: 'stay',
        label: 'Hoxton Paris family loft',
        url: 'https://thehoxton.com/paris/rooms/',
        details: 'Fits 2 adults + 1 child with breakfast add-on.',
      },
      {
        category: 'local_pass',
        label: 'Paris Museum Pass (4 days)',
        url: 'https://www.parismuseumpass.fr/t-en',
      },
      {
        category: 'sightseeing',
        label: 'Berlin Story Bunker timed ticket',
        url: 'https://www.getyourguide.com/berlin-story-museum-l4305/',
      },
    ],
  },
  baseline_plan: {
    query_echo: sampleRequest,
    options: [
      {
        label: 'balanced',
        summary:
          'High-speed trains between cities, boutique hotels near major sights, and prebooked passes keep the pace steady.',
        total_cost: 3985.4,
        currency: 'USD',
        transfers: 2,
        est_duration_days: 11,
        travel: [
          {
            mode: 'train',
            frm: 'Paris',
            to: 'Amsterdam',
            date: '2025-10-14',
            duration_hr: 3.25,
            cost_estimate: 115,
          },
          {
            mode: 'train',
            frm: 'Amsterdam',
            to: 'Berlin',
            date: '2025-10-17',
            duration_hr: 6.25,
            cost_estimate: 95,
          },
        ],
        stays: [
          { city: 'Paris', nights: 4, style: 'boutique', budget_per_night: 260 },
          { city: 'Amsterdam', nights: 3, style: 'boutique', budget_per_night: 245 },
          { city: 'Berlin', nights: 3, style: 'boutique', budget_per_night: 210 },
        ],
        local_transport: ['Navigo Easy', 'GVB day cards', 'Berlin WelcomeCard'],
        experience_plan: [
          {
            city: 'Paris',
            must_do: [
              'Timed Louvre entry + family highlights tour',
              'Evening Seine cruise with vegetarian dinner buffet',
            ],
            hidden_gem: ['Atelier des Lumières immersive light show'],
            flex_hours: 3,
          },
          {
            city: 'Amsterdam',
            must_do: ['Van Gogh Museum family guide', 'Private canal cruise at sunset'],
            hidden_gem: ['Jordaan food crawl focused on veggie bites'],
            flex_hours: 2,
          },
          {
            city: 'Berlin',
            must_do: ['Berlin Wall Memorial storyteller tour', 'Museum of Technology hands-on labs'],
            hidden_gem: ['Tempelhofer Feld picnic and kite session'],
            flex_hours: 2,
          },
        ],
        notes: [
          'Reserve key museum slots 60+ days ahead to secure family entry windows.',
          'Keep one evening free in each city for spontaneous finds or rest.',
        ],
        feasibility_notes: [
          'Rail timings assume morning departures with one-hour buffers at stations.',
          'All totals include vegetarian-friendly dining estimates at $160/day for the family.',
        ],
        transfer_buffers: {
          'Paris->Amsterdam': 1,
          'Amsterdam->Berlin': 1.5,
        },
        scores: {
          cost: 0.78,
          time: 0.82,
          experience: 0.9,
          composite: 0.84,
        },
        booking_links: [
          {
            category: 'train',
            label: 'Paris → Amsterdam Eurostar seats',
            url: 'https://www.eurostar.com/us-en/train/france/paris/paris-to-amsterdam',
          },
          {
            category: 'stay',
            label: 'Pulitzer Amsterdam connecting rooms',
            url: 'https://www.pulitzeramsterdam.com/rooms-suites/',
          },
          {
            category: 'sightseeing',
            label: 'Anne Frank House family entry',
            url: 'https://www.annefrank.org/en/museum/tickets/',
          },
        ],
      },
      {
        label: 'cheapest',
        summary: 'Self-catering apartments and slower regional trains stretch the budget.',
        total_cost: 3150,
        currency: 'USD',
        transfers: 2,
        est_duration_days: 11,
        travel: [
          {
            mode: 'train',
            frm: 'Paris',
            to: 'Amsterdam',
            date: '2025-10-14',
            duration_hr: 3.75,
            cost_estimate: 75,
          },
          {
            mode: 'train',
            frm: 'Amsterdam',
            to: 'Berlin',
            date: '2025-10-17',
            duration_hr: 6.5,
            cost_estimate: 69,
          },
        ],
        stays: [
          { city: 'Paris', nights: 4, style: 'apartment', budget_per_night: 180 },
          { city: 'Amsterdam', nights: 3, style: 'homestay', budget_per_night: 165 },
          { city: 'Berlin', nights: 3, style: 'apartment', budget_per_night: 150 },
        ],
        local_transport: ['Carnet metro tickets', 'OV-chipkaart', 'Berlin ABC transit pass'],
        experience_plan: [
          {
            city: 'Paris',
            must_do: ['Free walking tour of Île de la Cité', 'Picnic at Luxembourg Gardens'],
            hidden_gem: ['Canal Saint-Martin family paddle boats'],
            flex_hours: 4,
          },
          {
            city: 'Amsterdam',
            must_do: ['Cycling tour using MacBike rentals', 'Science day at NEMO museum'],
            hidden_gem: ['Tony\'s Chocolonely superstore tasting flight'],
            flex_hours: 3,
          },
          {
            city: 'Berlin',
            must_do: ['Free city walking tour from Brandenburg Gate', 'Exploratorium creative lab'],
            hidden_gem: ['Thaiwiese Sunday food market'],
            flex_hours: 3,
          },
        ],
        notes: [
          'Swap some restaurant meals for market picnics to stay on target.',
          'Regional train fares assume advance purchase saver tickets.',
        ],
        feasibility_notes: ['Apartments sourced within 15 minutes of central transit hubs.'],
        transfer_buffers: {
          'Paris->Amsterdam': 0.75,
          'Amsterdam->Berlin': 1,
        },
        scores: {
          cost: 0.92,
          time: 0.6,
          experience: 0.74,
          composite: 0.76,
        },
        booking_links: [
          {
            category: 'stay',
            label: 'Paris Citadines République apartment',
            url: 'https://www.discoverasr.com/en/citadines/france/citadines-republique-paris',
          },
          {
            category: 'local_pass',
            label: 'Amsterdam Go City pass (2-day)',
            url: 'https://gocity.com/amsterdam/en-us/products/explorer',
          },
          {
            category: 'bus',
            label: 'FlixBus fallback Amsterdam ↔ Berlin',
            url: 'https://www.flixbus.com/bus/amsterdam/berlin',
          },
        ],
      },
      {
        label: 'comfort',
        summary: 'Business-class flights, five-star suites, and private guides across all three cities.',
        total_cost: 6125,
        currency: 'USD',
        transfers: 2,
        est_duration_days: 11,
        travel: [
          {
            mode: 'flight',
            frm: 'Paris',
            to: 'Amsterdam',
            date: '2025-10-14',
            duration_hr: 1.1,
            cost_estimate: 240,
          },
          {
            mode: 'flight',
            frm: 'Amsterdam',
            to: 'Berlin',
            date: '2025-10-17',
            duration_hr: 1.3,
            cost_estimate: 260,
          },
        ],
        stays: [
          { city: 'Paris', nights: 4, style: 'boutique', budget_per_night: 420 },
          { city: 'Amsterdam', nights: 3, style: 'boutique', budget_per_night: 410 },
          { city: 'Berlin', nights: 3, style: 'boutique', budget_per_night: 360 },
        ],
        local_transport: ['Private car service', 'Canal boat with skipper', 'Private driver'],
        experience_plan: [
          {
            city: 'Paris',
            must_do: ['Private Louvre before-hours tour', 'Chef-led cooking class with market visit'],
            hidden_gem: ['Champagne day trip with driver'],
            flex_hours: 2,
          },
          {
            city: 'Amsterdam',
            must_do: ['Exclusive Rijksmuseum after-hours tour', 'Windmill countryside helicopter hop'],
            hidden_gem: ['Micropia scientist meet-and-greet'],
            flex_hours: 2,
          },
          {
            city: 'Berlin',
            must_do: ['Private Third Reich history guide', 'Street art workshop in Friedrichshain'],
            hidden_gem: ['Chef\'s table at Cookies Cream vegetarian fine dining'],
            flex_hours: 1,
          },
        ],
        notes: ['Daily spa access and childcare add-ons baked into totals.'],
        feasibility_notes: [
          'Premium flight fares from SkyTeam carriers with through-check baggage.',
          'Private guides sourced through Virtuoso partners with family credentials.',
        ],
        transfer_buffers: {
          'Paris->Amsterdam': 2,
          'Amsterdam->Berlin': 2,
        },
        scores: {
          cost: 0.52,
          time: 0.95,
          experience: 0.98,
          composite: 0.86,
        },
        booking_links: [
          {
            category: 'flight',
            label: 'Air France Business Flex SFO ↔ CDG',
            url: 'https://www.airfrance.us/plan-your-trip/book-a-flight',
          },
          {
            category: 'stay',
            label: 'Waldorf Astoria Amsterdam two-bedroom suite',
            url: 'https://www.hilton.com/en/hotels/amstwwa-waldorf-astoria-amsterdam/rooms/',
          },
          {
            category: 'sightseeing',
            label: 'Berlin private guide collective',
            url: 'https://www.toursbylocals.com/Berlin-Tours',
          },
        ],
      },
    ],
  },
  agent_context: {
    notes: [
      'Live pricing to be refreshed within 24h of booking to capture fare shifts.',
      'LLM (gpt-4o-mini) supplied supplemental descriptions where live snippets were unavailable.',
    ],
    sources: [
      {
        city: 'Paris',
        title: 'Paris Museum Pass official site',
        url: 'https://www.parismuseumpass.fr/t-en',
      },
      {
        city: 'Amsterdam',
        title: 'GVB public transport day tickets',
        url: 'https://www.gvb.nl/en/tickets',
      },
      {
        city: 'Berlin',
        title: 'Deutsche Bahn saver fares overview',
        url: 'https://www.bahn.com/en/offers/long-distance/saver-fares-europe',
      },
    ],
    llm_sources: [
      {
        model: 'gpt-4o-mini',
        reason: 'Filled in attraction blurbs when scraping fell back to heuristics.',
      },
    ],
  },
};
