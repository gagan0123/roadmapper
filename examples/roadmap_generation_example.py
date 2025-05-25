#!/usr/bin/env python3
"""
Roadmap Generation System Example

This example demonstrates how to use the complete agentic roadmap generation system
with all four agents working together to create comprehensive roadmaps.
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any

# Add the root directory to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import shared models and flow
from shared.models.roadmap_models import (
    FlowInput, DocumentSource, SourceType, PriorityLevel
)
from agentic_flow.flows.roadmap_generation_flow import generate_roadmap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_full_roadmap_generation():
    """
    Demonstrate the complete roadmap generation workflow.
    
    This example shows how all four agents work together:
    1. Document Loader: Processes strategic documents
    2. Strategy Aligner: Analyzes strategic context
    3. Ideation: Generates innovative ideas
    4. Roadmap Synthesis: Creates comprehensive roadmaps
    """
    
    logger.info("🚀 Starting Roadmap Generation System Demo")
    logger.info("="*80)
    
    # Configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return
    
    try:
        # Step 1: Configure input sources and objectives
        logger.info("Step 1: Configuring input sources and strategic objectives...")
        
        # Define document sources
        sources = [
            DocumentSource(
                type=SourceType.WEB,
                config={
                    "url": "https://developers.googleblog.com/en/agents-adk-agent-engine-a2a-enhancements-google-io/"
                },
                priority=PriorityLevel.HIGH,
                metadata={"category": "technology_strategy"}
            ),
            DocumentSource(
                type=SourceType.WEB,
                config={
                    "url": "https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/"
                },
                priority=PriorityLevel.HIGH,
                metadata={"category": "agentic_systems"}
            )
        ]
        
        # Define strategic objectives
        objectives = [
            "Develop next-generation AI-powered products",
            "Improve customer experience through intelligent automation",
            "Enhance operational efficiency with agentic workflows",
            "Build scalable multi-agent systems",
            "Establish market leadership in AI innovation"
        ]
        
        # Define constraints and preferences
        constraints = [
            "12-month initial timeline",
            "Limited budget for external vendors",
            "Compliance with data privacy regulations",
            "Integration with existing systems required"
        ]
        
        # Create flow input
        flow_input = FlowInput(
            project_id=project_id,
            sources=sources,
            objectives=objectives,
            constraints=constraints,
            timeline_preferences={
                "duration_months": 18,
                "phases": 3,
                "quick_wins_timeline": "Q1"
            },
            resource_constraints={
                "budget_range": "medium",
                "team_size": "10-15",
                "external_dependencies": "minimal"
            },
            preferences={
                "innovation_level": "high",
                "risk_tolerance": "medium",
                "strategic_focus": ["ai_capabilities", "customer_value", "operational_excellence"]
            }
        )
        
        logger.info(f"✓ Configured {len(sources)} document sources")
        logger.info(f"✓ Defined {len(objectives)} strategic objectives")
        logger.info(f"✓ Set {len(constraints)} constraints")
        
        # Step 2: Execute the roadmap generation flow
        logger.info("\nStep 2: Executing roadmap generation flow...")
        logger.info("This coordinates all four agents in sequence:")
        logger.info("  📄 Document Loader → 🎯 Strategy Aligner → 💡 Ideation → 🗺️ Roadmap Synthesis")
        
        # Convert to dict for the flow function
        input_data = {
            "project_id": flow_input.project_id,
            "sources": [source.dict() for source in flow_input.sources],
            "objectives": flow_input.objectives,
            "constraints": flow_input.constraints,
            "timeline_preferences": flow_input.timeline_preferences,
            "resource_constraints": flow_input.resource_constraints,
            "preferences": flow_input.preferences,
            "use_vector_search": True
        }
        
        # Execute the flow
        result = await generate_roadmap(input_data)
        
        # Step 3: Display results
        logger.info("\nStep 3: Roadmap Generation Results")
        logger.info("="*50)
        
        if result.get("status") == "success":
            logger.info("✅ Roadmap generation completed successfully!")
            
            # Document processing results
            doc_count = result.get("documents", 0)
            logger.info(f"📄 Documents processed: {doc_count}")
            
            # Strategic analysis results
            strategic_context = result.get("strategic_context", {})
            key_themes = strategic_context.get("key_themes", [])
            alignment_score = strategic_context.get("alignment_score", 0)
            logger.info(f"🎯 Strategic themes identified: {', '.join(key_themes)}")
            logger.info(f"🎯 Strategic alignment score: {alignment_score:.2f}")
            
            # Ideation results
            ideas_count = result.get("ideas", 0)
            logger.info(f"💡 Ideas generated: {ideas_count}")
            
            # Roadmap results
            roadmap = result.get("roadmap", {})
            roadmap_title = roadmap.get("title", "Strategic Roadmap")
            timeline = roadmap.get("timeline", {})
            phases = timeline.get("phases", [])
            success_metrics = roadmap.get("success_metrics", [])
            
            logger.info(f"\n🗺️ Generated Roadmap: {roadmap_title}")
            logger.info(f"📅 Timeline: {timeline.get('start_date')} to {timeline.get('end_date')}")
            logger.info(f"🔄 Phases: {len(phases)}")
            
            for i, phase in enumerate(phases, 1):
                logger.info(f"   Phase {i}: {phase.get('name')} ({phase.get('duration')})")
                initiatives = phase.get('initiatives', [])
                deliverables = phase.get('deliverables', [])
                logger.info(f"     - Initiatives: {len(initiatives)}")
                logger.info(f"     - Deliverables: {', '.join(deliverables[:2])}{'...' if len(deliverables) > 2 else ''}")
            
            logger.info(f"\n📊 Success Metrics:")
            for metric in success_metrics:
                logger.info(f"   • {metric}")
            
            # Priorities breakdown
            priorities = roadmap.get("priorities", {})
            high_priority = priorities.get("high", [])
            medium_priority = priorities.get("medium", [])
            low_priority = priorities.get("low", [])
            
            logger.info(f"\n🎯 Priority Breakdown:")
            logger.info(f"   High: {len(high_priority)} initiatives")
            logger.info(f"   Medium: {len(medium_priority)} initiatives")
            logger.info(f"   Low: {len(low_priority)} initiatives")
            
            # Metadata
            metadata = result.get("metadata", {})
            agents_used = metadata.get("agents_used", [])
            flow_version = metadata.get("flow_version", "unknown")
            
            logger.info(f"\n🔧 Flow Metadata:")
            logger.info(f"   Flow version: {flow_version}")
            logger.info(f"   Agents used: {', '.join(agents_used)}")
            
        else:
            logger.error("❌ Roadmap generation failed!")
            error = result.get("error", "Unknown error")
            logger.error(f"Error: {error}")
            
            # Show any partial results
            partial_results = result.get("partial_results", {})
            if partial_results:
                logger.info("Partial results available:")
                for key, value in partial_results.items():
                    logger.info(f"  {key}: {value}")
        
        # Step 4: Next steps and recommendations
        logger.info("\n" + "="*80)
        logger.info("🎯 NEXT STEPS")
        logger.info("="*80)
        logger.info("1. Review the generated roadmap for strategic alignment")
        logger.info("2. Validate ideas with stakeholders and domain experts")  
        logger.info("3. Refine timelines based on resource availability")
        logger.info("4. Implement tracking mechanisms for success metrics")
        logger.info("5. Set up regular roadmap review and update cycles")
        
        logger.info("\n🔧 CUSTOMIZATION OPTIONS")
        logger.info("- Adjust agent configurations for different domains")
        logger.info("- Add custom document sources and strategic frameworks")
        logger.info("- Integrate with project management and tracking tools")
        logger.info("- Customize output formats and visualization options")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error("Check your configuration and try again")
        raise


async def demonstrate_individual_agent_development():
    """
    Show how individual agents can still be developed and tested independently.
    """
    logger.info("\n" + "="*80) 
    logger.info("🔧 INDIVIDUAL AGENT DEVELOPMENT")
    logger.info("="*80)
    
    logger.info("Each agent can be developed and tested independently:")
    
    logger.info("\n📄 Document Loader Agent:")
    logger.info("   cd agents/document_loader")
    logger.info("   python -m pytest tests/")
    logger.info("   python examples/vector_search_example.py")
    
    logger.info("\n🎯 Strategy Aligner Agent:")
    logger.info("   cd agents/strategy_aligner")
    logger.info("   python -m pytest tests/")
    logger.info("   python examples/strategy_alignment_example.py")
    
    logger.info("\n💡 Ideation Agent:")
    logger.info("   cd agents/ideation")
    logger.info("   python -m pytest tests/")
    logger.info("   python examples/idea_generation_example.py")
    
    logger.info("\n🗺️ Roadmap Synthesis Agent:")
    logger.info("   cd agents/roadmap_synthesis")
    logger.info("   python -m pytest tests/")
    logger.info("   python examples/roadmap_synthesis_example.py")
    
    logger.info("\n🔗 Integration Testing:")
    logger.info("   python -m pytest tests/integration/")
    logger.info("   python -m pytest tests/e2e/")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Roadmap Generation System Demo")
    parser.add_argument("--full-demo", action="store_true", 
                       help="Run the complete roadmap generation demo")
    parser.add_argument("--show-dev-workflow", action="store_true",
                       help="Show individual agent development workflow")
    
    args = parser.parse_args()
    
    if args.full_demo:
        asyncio.run(demonstrate_full_roadmap_generation())
    elif args.show_dev_workflow:
        asyncio.run(demonstrate_individual_agent_development())
    else:
        # Run both by default
        asyncio.run(demonstrate_full_roadmap_generation())
        asyncio.run(demonstrate_individual_agent_development())


if __name__ == "__main__":
    main() 