"""
Profile Updater Utility.

Handles updating YAML configuration profiles (e.g., adding auto-extracted anchor terms).
"""

import os
import yaml
import logging
from typing import List, Any, Dict

logger = logging.getLogger(__name__)

class ProfileUpdater:
    """
    Utility for modifying KB configuration profiles.
    """

    @staticmethod
    def update_collection_anchor_terms(profile_path: str, terms: List[str]) -> bool:
        """
        Update the collection_anchor_terms in a YAML profile.

        Args:
            profile_path: Path to the YAML file
            terms: List of new anchor terms

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(profile_path):
            logger.error(f"Profile path does not exist: {profile_path}")
            return False

        try:
            # Load the YAML
            with open(profile_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            # Ensure grounding section exists
            if 'grounding' not in config:
                config['grounding'] = {}
            
            # Update terms
            # Merge with existing terms if desired, or overwrite?
            # The diagram says "Update collection_anchor_terms in profile", 
            # usually overwrite is safer for "auto-populate" during index.
            config['grounding']['collection_anchor_terms'] = terms

            # Save back to file
            with open(profile_path, 'w') as f:
                # Use default_flow_style=None to keep lists readable
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Successfully updated anchor terms in {profile_path}: {terms}")
            return True

        except Exception as e:
            logger.error(f"Failed to update profile {profile_path}: {e}")
            return False

    @staticmethod
    def update_kb_scope_description(profile_path: str, scope_description: str) -> bool:
        """
        Update the kb_scope_description in a YAML profile.

        Args:
            profile_path: Path to the YAML file
            scope_description: The generated scope paragraph

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(profile_path):
            logger.error(f"Profile path does not exist: {profile_path}")
            return False

        try:
            # Load the YAML
            with open(profile_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            # Ensure grounding section exists
            if 'grounding' not in config:
                config['grounding'] = {}
            
            # Update scope description
            config['grounding']['kb_scope_description'] = scope_description

            # NEW: Automatically ensure the prompt template includes the grounding block
            updated_template = False
            if 'prompt' in config and 'template' in config['prompt']:
                template = config['prompt']['template']
                if 'kb_scope_description' not in template:
                    logger.info("Injecting grounding block into prompt template for %s", profile_path)
                    
                    grounding_block = (
                        "\n\n    {% if kb_scope_description %}\n"
                        "    Knowledge Base Scope:\n"
                        "    {{ kb_scope_description }}\n"
                        "    {% endif %}"
                    )
                    
                    # Try to find a good injection point (Tenant Instructions)
                    target = "knowledge base content first."
                    if target in template:
                        parts = template.split(target, 1)
                        config['prompt']['template'] = parts[0] + target + grounding_block + parts[1]
                        updated_template = True
                    else:
                        # Fallback: find any instruction section
                        fallback_target = "### Tenant Instructions (CRITICAL)"
                        if fallback_target in template:
                            parts = template.split(fallback_target, 1)
                            config['prompt']['template'] = parts[0] + fallback_target + grounding_block + parts[1]
                            updated_template = True

            # Save back to file
            with open(profile_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            if updated_template:
                logger.info(f"Successfully updated scope description and prompt template in {profile_path}")
            else:
                logger.info(f"Successfully updated scope description in {profile_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to update profile scope {profile_path}: {e}")
            return False

    @staticmethod
    def get_collection_anchor_terms(profile_path: str) -> List[str]:
        """Get existing anchor terms from a profile."""
        if not os.path.exists(profile_path):
            return []
            
        try:
            with open(profile_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return config.get('grounding', {}).get('collection_anchor_terms', [])
        except Exception:
            return []
