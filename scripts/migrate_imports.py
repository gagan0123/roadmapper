#!/usr/bin/env python3
"""
Import Migration Script

This script helps update import paths after migrating to the monorepo structure.
It updates imports in the Document Loader agent to work with the new structure.
"""

import os
import re
import glob
from pathlib import Path


def update_imports_in_file(file_path: str) -> bool:
    """Update import statements in a single file."""
    
    # Import patterns to update
    patterns = [
        # Update src.* imports to agents.document_loader.src.*
        (r'from src\.', 'from agents.document_loader.src.'),
        (r'import src\.', 'import agents.document_loader.src.'),
        
        # Update relative imports within document loader
        (r'from \.\.src\.', 'from agents.document_loader.src.'),
        (r'from \.src\.', 'from agents.document_loader.src.'),
        
        # Update @patch decorators in test files - THIS IS THE KEY FIX!
        (r"@patch\('src\.", "@patch('agents.document_loader.src."),
        (r'@patch\("src\.', '@patch("agents.document_loader.src.'),
        
        # Update assertLogs calls in test files
        (r"logger='src\.", "logger='agents.document_loader.src."),
        (r'logger="src\.', 'logger="agents.document_loader.src.'),
        
        # Update patch calls inside with statements
        (r"patch\('src\.", "patch('agents.document_loader.src."),
        (r'patch\("src\.', 'patch("agents.document_loader.src.'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply each pattern
        for old_pattern, new_pattern in patterns:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False


def update_document_loader_imports():
    """Update imports in the Document Loader agent files."""
    
    print("🔄 Updating Document Loader imports...")
    
    # Directories to process
    directories = [
        "agents/document_loader/src",
        "agents/document_loader/tests",
        "agents/document_loader/examples"
    ]
    
    total_files = 0
    updated_files = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            continue
            
        # Find all Python files
        python_files = glob.glob(f"{directory}/**/*.py", recursive=True)
        
        for file_path in python_files:
            total_files += 1
            if update_imports_in_file(file_path):
                updated_files += 1
                print(f"✓ Updated: {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files updated: {updated_files}")
    print(f"   Files unchanged: {total_files - updated_files}")


def create_development_scripts():
    """Create convenience scripts for development workflow."""
    
    print("\n🛠️  Creating development scripts...")
    
    # Create scripts directory
    os.makedirs("scripts", exist_ok=True)
    
    # Test script for individual agents
    test_script = """#!/bin/bash
# Test individual agents

set -e

echo "🧪 Testing individual agents..."

echo "📄 Testing Document Loader Agent..."
cd agents/document_loader && python -m pytest tests/ -v
cd ../..

echo "🎯 Testing Strategy Aligner Agent..."
cd agents/strategy_aligner && python -m pytest tests/ -v
cd ../..

echo "💡 Testing Ideation Agent..."
cd agents/ideation && python -m pytest tests/ -v
cd ../..

echo "🗺️ Testing Roadmap Synthesis Agent..."
cd agents/roadmap_synthesis && python -m pytest tests/ -v
cd ../..

echo "✅ All agent tests completed!"
"""
    
    with open("scripts/test_agents.sh", "w") as f:
        f.write(test_script)
    os.chmod("scripts/test_agents.sh", 0o755)
    
    # Integration test script
    integration_script = """#!/bin/bash
# Test integration and end-to-end workflows

set -e

echo "🔗 Running integration tests..."
python -m pytest tests/integration/ -v

echo "🌐 Running end-to-end tests..."
python -m pytest tests/e2e/ -v

echo "✅ Integration tests completed!"
"""
    
    with open("scripts/test_integration.sh", "w") as f:
        f.write(integration_script)
    os.chmod("scripts/test_integration.sh", 0o755)
    
    # Example runner script
    example_script = """#!/bin/bash
# Run examples for all agents

set -e

echo "🎮 Running agent examples..."

echo "📄 Document Loader Example..."
cd agents/document_loader && python examples/vector_search_example.py
cd ../..

echo "🗺️ Full Roadmap Generation Example..."
python examples/roadmap_generation_example.py --full-demo

echo "✅ All examples completed!"
"""
    
    with open("scripts/run_examples.sh", "w") as f:
        f.write(example_script)
    os.chmod("scripts/run_examples.sh", 0o755)
    
    # Quick test script for Document Loader only
    doc_loader_test_script = """#!/bin/bash
# Quick test script for Document Loader agent only

set -e

echo "📄 Testing Document Loader Agent..."
cd agents/document_loader
echo "Running unit tests..."
python -m pytest tests/ -v -x
echo "✅ Document Loader tests completed!"
"""
    
    with open("scripts/test_doc_loader.sh", "w") as f:
        f.write(doc_loader_test_script)
    os.chmod("scripts/test_doc_loader.sh", 0o755)
    
    print("✓ Created scripts/test_agents.sh")
    print("✓ Created scripts/test_integration.sh") 
    print("✓ Created scripts/run_examples.sh")
    print("✓ Created scripts/test_doc_loader.sh")


def verify_migration():
    """Verify the migration was successful."""
    
    print("\n🔍 Verifying migration...")
    
    # Check that key directories exist
    required_dirs = [
        "agents/document_loader/src",
        "agents/document_loader/tests",
        "agents/document_loader/examples",
        "agents/strategy_aligner",
        "agents/ideation", 
        "agents/roadmap_synthesis",
        "agentic_flow/flows",
        "shared/models",
        "deployment"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
        else:
            print(f"✓ {directory}")
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories:")
        for directory in missing_dirs:
            print(f"   ❌ {directory}")
        return False
    
    # Check that key files exist
    required_files = [
        "agents/document_loader/README.md",
        "shared/models/roadmap_models.py",
        "agentic_flow/flows/roadmap_generation_flow.py",
        "examples/roadmap_generation_example.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for file_path in missing_files:
            print(f"   ❌ {file_path}")
        return False
    
    print("\n✅ Migration verification completed successfully!")
    return True


def verify_import_fixes():
    """Verify that import fixes have been applied correctly."""
    
    print("\n🔍 Verifying import fixes...")
    
    # Check a few key test files for proper @patch paths
    test_files = [
        "agents/document_loader/tests/test_vector_search_index_manager.py",
        "agents/document_loader/tests/test_drive_loader.py",
        "agents/document_loader/tests/test_web_loader.py"
    ]
    
    fixes_needed = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
                
            # Check for old-style imports that should have been fixed
            old_patterns = [
                r"@patch\('src\.",
                r'@patch\("src\.',
                r'from src\.',
                r'import src\.'
            ]
            
            for pattern in old_patterns:
                if re.search(pattern, content):
                    fixes_needed.append(f"{test_file}: Found {pattern}")
                    break
            else:
                print(f"✓ {test_file}")
        else:
            print(f"⚠️  File not found: {test_file}")
    
    if fixes_needed:
        print(f"\n⚠️  Import fixes still needed:")
        for fix in fixes_needed:
            print(f"   ❌ {fix}")
        return False
    
    print("✅ All import fixes verified!")
    return True


def main():
    """Main migration function."""
    
    print("🚀 Starting monorepo migration...")
    print("="*60)
    
    # Step 1: Update imports
    update_document_loader_imports()
    
    # Step 2: Create development scripts
    create_development_scripts()
    
    # Step 3: Verify migration
    migration_ok = verify_migration()
    imports_ok = verify_import_fixes()
    
    if migration_ok and imports_ok:
        print("\n" + "="*60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎯 Next steps:")
        print("1. Test Document Loader agent: ./scripts/test_doc_loader.sh")
        print("2. Run full system example: python examples/roadmap_generation_example.py")
        print("3. Start developing other agents using the established patterns")
        print("4. Use scripts/test_agents.sh for convenient testing")
        print("\n🔧 Development workflow:")
        print("- Each agent can be developed independently")
        print("- Shared models are available in shared/models/")
        print("- ADK orchestration is in agentic_flow/")
        print("- Use scripts/ for common development tasks")
        
        print("\n🧪 Quick test command:")
        print("cd agents/document_loader && python -m pytest tests/test_vector_search_index_manager.py::TestVectorSearchIndexManagerInitialization::test_successful_initialization -v")
    else:
        print("\n❌ Migration verification failed. Please check the issues above.")


if __name__ == "__main__":
    main() 