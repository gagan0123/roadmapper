from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import shlex
import logging # Import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Or your desired default level

VALID_MODEL_TYPES = ["text", "multimodal"]

class DataSourceLoader:
    """Loads data source configurations from text files."""
    
    def __init__(self, sources_dir: str = "data/sources"):
        self.sources_dir = Path(sources_dir)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.default_model_type = "multimodal"
        
        # Create default files if they don't exist
        self._create_default_files()
    
    def _create_default_files(self):
        """Create default source files if they don't exist."""
        default_files = {
            'drive_folders.txt': '# One folder ID per line, optionally followed by a category, then model type (text/multimodal).\n# Example: 1abc...xyz StrategyDoc text\n# Example: 2def...uvw "Product Design" multimodal\n# Example: 3ghi...jkl SomeDocs\n',
            'github_repos.txt': '# One repository per line (owner/repo), optionally followed by a category, then model type.\n# Example: google/vertex-ai-samples CodeBase text\n# Example: myorg/myrepo "Internal Tools" multimodal\n# Example: otherorg/another\n',
            'web_urls.txt': '# One URL per line, optionally followed by a category, then model type.\n# Example: https://example.com Blog text\n# Example: https://another.com/page "Research Papers" multimodal\n# Example: https://yet.another.com/article\n'
        }
        
        for filename, content in default_files.items():
            file_path = self.sources_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(content)
    
    def _parse_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a single line from a data source file."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None # Skip empty lines and comments

        parts = shlex.split(line)

        if not parts: # Should be caught by 'if not line' but as a safeguard
            return None, None, None

        item_name = parts[0] if len(parts) > 0 else None
        category = parts[1] if len(parts) > 1 else None
        model_type_str = parts[2] if len(parts) > 2 else None

        if not item_name: # If, after shlex, item_name is somehow empty or None
            return None, None, None

        final_model_type = self.default_model_type
        if model_type_str:
            if model_type_str.lower() in VALID_MODEL_TYPES:
                final_model_type = model_type_str.lower()
            else:
                logger.warning(f"Invalid model type '{model_type_str}' in line: '{line}'. Defaulting to '{self.default_model_type}'.")
        # If model_type_str is None, final_model_type remains self.default_model_type

        return item_name, category, final_model_type

    def _load_file_content(self, file_path: Path) -> List[Tuple[str, Optional[str], str]]:
        """Load and parse content from a file, expecting item, optional category, and model type per line."""
        if not file_path.exists():
            return []
        
        parsed_items = []
        with open(file_path, 'r') as f:
            for line_content in f.readlines():
                item, category, model_type = self._parse_line(line_content)
                if item: # Only add if an item was successfully parsed
                    parsed_items.append((item, category, model_type))
        return parsed_items
    
    def load_drive_folders(self) -> List[Tuple[str, Optional[str], str]]:
        """Load Google Drive folder IDs, their categories, and model types from file."""
        file_path = self.sources_dir / 'drive_folders.txt'
        return self._load_file_content(file_path)
    
    def load_github_repos(self) -> List[Tuple[str, Optional[str], str]]:
        """Load GitHub repository names, their categories, and model types from file."""
        file_path = self.sources_dir / 'github_repos.txt'
        return self._load_file_content(file_path)
    
    def load_web_urls(self) -> List[Tuple[str, Optional[str], str]]:
        """Load web URLs, their categories, and model types from file."""
        file_path = self.sources_dir / 'web_urls.txt'
        return self._load_file_content(file_path)
    
    def get_all_sources(self) -> Dict[str, List[Tuple[str, Optional[str], str]]]:
        """Get all data sources as a dictionary, with items, their categories, and model types."""
        return {
            'drive_folders': self.load_drive_folders(),
            'github_repos': self.load_github_repos(),
            'web_urls': self.load_web_urls()
        } 