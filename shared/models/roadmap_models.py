"""
Shared data models for the Roadmap Generation Agentic System.

These models define the common data structures used across all agents
and the agentic flow orchestration.
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class PriorityLevel(str, Enum):
    """Priority levels for ideas and initiatives."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EffortLevel(str, Enum):
    """Effort levels for implementation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactLevel(str, Enum):
    """Impact levels for business value."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentType(str, Enum):
    """Types of agents in the system."""
    DOCUMENT_LOADER = "document_loader"
    STRATEGY_ALIGNER = "strategy_aligner"
    IDEATION = "ideation"
    ROADMAP_SYNTHESIS = "roadmap_synthesis"


class SourceType(str, Enum):
    """Types of document sources."""
    WEB = "web"
    GITHUB = "github"
    DRIVE = "drive"
    LOCAL = "local"


# Base Models
class BaseAgentOutput(BaseModel):
    """Base model for all agent outputs."""
    agent_type: AgentType
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    status: str = "success"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSource(BaseModel):
    """Configuration for a document source."""
    type: SourceType
    config: Dict[str, Any]
    priority: PriorityLevel = PriorityLevel.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Document Loader Models
class ProcessedDocument(BaseModel):
    """Processed document from Document Loader Agent."""
    id: str
    title: str
    content: str
    source_type: SourceType
    source_url: Optional[str] = None
    category: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.now)


class DocumentLoaderOutput(BaseAgentOutput):
    """Output from Document Loader Agent."""
    agent_type: AgentType = AgentType.DOCUMENT_LOADER
    documents: List[ProcessedDocument]
    total_documents: int
    sources_processed: List[str]
    embedding_stats: Dict[str, Any] = Field(default_factory=dict)


# Strategy Aligner Models
class StrategicObjective(BaseModel):
    """A strategic objective identified by the Strategy Aligner Agent."""
    id: str
    title: str
    description: str
    category: str
    priority: PriorityLevel
    success_criteria: List[str]
    constraints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategicAlignment(BaseModel):
    """Strategic alignment analysis result."""
    objective_id: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_evidence: List[str] = Field(default_factory=list)
    gaps_identified: List[str] = Field(default_factory=list)


class StrategyAlignerOutput(BaseAgentOutput):
    """Output from Strategy Aligner Agent."""
    agent_type: AgentType = AgentType.STRATEGY_ALIGNER
    objectives: List[StrategicObjective]
    key_themes: List[str]
    constraints: List[str]
    alignments: List[StrategicAlignment]
    overall_alignment_score: float = Field(ge=0.0, le=1.0)


# Ideation Models
class GeneratedIdea(BaseModel):
    """An idea generated by the Ideation Agent."""
    id: str
    title: str
    description: str
    category: str
    priority: PriorityLevel
    effort: EffortLevel
    impact: ImpactLevel
    alignment_score: float = Field(ge=0.0, le=1.0)
    supporting_documents: List[str] = Field(default_factory=list)
    related_objectives: List[str] = Field(default_factory=list)
    implementation_notes: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdeationOutput(BaseAgentOutput):
    """Output from Ideation Agent."""
    agent_type: AgentType = AgentType.IDEATION
    ideas: List[GeneratedIdea]
    total_ideas: int
    categories: List[str]
    innovation_metrics: Dict[str, Any] = Field(default_factory=dict)


# Roadmap Synthesis Models
class RoadmapPhase(BaseModel):
    """A phase in the roadmap timeline."""
    name: str
    start_date: date
    end_date: date
    duration: str
    initiatives: List[str]  # IDs of ideas/initiatives in this phase
    deliverables: List[str]
    dependencies: List[str] = Field(default_factory=list)
    resources_required: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)


class RoadmapTimeline(BaseModel):
    """Timeline structure for the roadmap."""
    start_date: date
    end_date: date
    phases: List[RoadmapPhase]
    milestones: List[str] = Field(default_factory=list)


class PrioritizedInitiatives(BaseModel):
    """Prioritized initiatives organized by priority level."""
    high: List[GeneratedIdea]
    medium: List[GeneratedIdea]
    low: List[GeneratedIdea]


class GeneratedRoadmap(BaseModel):
    """The final generated roadmap."""
    title: str
    description: str
    timeline: RoadmapTimeline
    priorities: PrioritizedInitiatives
    success_metrics: List[str]
    risk_factors: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    total_budget_estimate: Optional[float] = None
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)


class RoadmapSynthesisOutput(BaseAgentOutput):
    """Output from Roadmap Synthesis Agent."""
    agent_type: AgentType = AgentType.ROADMAP_SYNTHESIS
    roadmap: GeneratedRoadmap
    synthesis_summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)


# Agentic Flow Models
class FlowInput(BaseModel):
    """Input configuration for the roadmap generation flow."""
    project_id: str
    sources: List[DocumentSource]
    objectives: List[str]
    constraints: List[str] = Field(default_factory=list)
    timeline_preferences: Dict[str, Any] = Field(default_factory=dict)
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class FlowOutput(BaseModel):
    """Complete output from the roadmap generation flow."""
    status: str
    flow_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Agent outputs
    document_loader_output: Optional[DocumentLoaderOutput] = None
    strategy_aligner_output: Optional[StrategyAlignerOutput] = None
    ideation_output: Optional[IdeationOutput] = None
    roadmap_synthesis_output: Optional[RoadmapSynthesisOutput] = None
    
    # Summary metrics
    total_documents_processed: int = 0
    total_ideas_generated: int = 0
    overall_confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    agents_used: List[AgentType] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Validation helpers
class ModelValidationMixin:
    """Mixin for common model validations."""
    
    @validator('alignment_score', 'confidence_score', 'overall_alignment_score', pre=True, always=True)
    def validate_score_range(cls, v):
        """Ensure scores are between 0.0 and 1.0."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Score must be between 0.0 and 1.0')
        return v 