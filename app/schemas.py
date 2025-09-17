from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, AliasChoices, ConfigDict

# ------- Request models -------
class TripPrefs(BaseModel):
    objective: Literal["balanced", "cheapest", "comfort", "family_friendly"] = "balanced"
    flexible_days: int = 0
    max_flight_hours: Optional[float] = None
    diet: List[str] = Field(default_factory=list)  # type: ignore
    mobility: Literal["normal", "step_free", "low_stairs"] = "normal"

class Party(BaseModel):
    adults: int = 2
    children: int = 0
    seniors: int = 0

class Dates(BaseModel):
    start: str
    end: str

class TripRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    origin: str
    destinations: List[str]
    dates: Dates
    budget_total: float = Field(..., validation_alias=AliasChoices("budget_total", "budget"))
    currency: str = "USD"
    party: Party = Party()
    prefs: TripPrefs = TripPrefs()

# ------- Response models -------
class Stay(BaseModel):
    city: str
    nights: int
    style: Literal["hotel", "boutique", "homestay", "apartment"] = "hotel"
    budget_per_night: float

class TravelLeg(BaseModel):
    mode: Literal["flight","train","bus","car","rideshare","metro","walk","ferry"]
    frm: str = Field(..., alias="from")
    to: str
    date: Optional[str] = None
    duration_hr: Optional[float] = None
    cost_estimate: Optional[float] = None

class DayPlan(BaseModel):
    city: str
    must_do: List[str] = Field(default_factory=list)       # [] -> default_factory
    hidden_gem: List[str] = Field(default_factory=list)    # [] -> default_factory
    flex_hours: int = 2

class AgentContext(BaseModel):
    foundation: Dict[str, object] = Field(default_factory=dict)
    destinations: List[Dict[str, object]] = Field(default_factory=list)
    logistics: Dict[str, object] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    snippets: List[Dict[str, str]] = Field(default_factory=list)

class PlanBundle(BaseModel):
    label: Literal["balanced","cheapest","comfort","family_friendly"]
    summary: str
    total_cost: float
    currency: str
    transfers: int
    est_duration_days: int
    travel: List[TravelLeg]
    stays: List[Stay]
    local_transport: List[str] = Field(default_factory=list)    # [] -> default_factory
    experience_plan: List[DayPlan] = Field(default_factory=list) # [] -> default_factory
    notes: List[str] = Field(default_factory=list)               # [] -> default_factory
    feasibility_notes: List[str] = Field(default_factory=list)
    transfer_buffers: Dict[str, float] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)

class TripResponse(BaseModel):
    query_echo: TripRequest
    options: List[PlanBundle] = Field(default_factory=list)      # [] -> default_factory
    agent_context: Optional[AgentContext] = None
