#!/usr/bin/env python3
"""
Dependency update utilities for RiskX platform.
Manages Python and Node.js dependency updates with security checking.
"""

import os
import sys
import json
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class DependencyUpdater:
    """Dependency management and updating for RiskX platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.requirements_dev_file = self.project_root / "requirements-dev.txt"
        self.package_json = self.project_root / "package.json"
        self.frontend_package_json = self.project_root / "frontend" / "package.json"
        
        self.update_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "python_updates": [],
            "node_updates": [],
            "security_issues": [],
            "errors": []
        }
    
    def check_python_security(self) -> List[Dict[str, Any]]:
        """Check Python dependencies for security vulnerabilities."""
        security_issues = []
        
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0 and result.stdout:
                try:
                    issues = json.loads(result.stdout)
                    for issue in issues:
                        security_issues.append({
                            "type": "python",
                            "package": issue.get("package", "unknown"),
                            "vulnerability": issue.get("vulnerability", "unknown"),
                            "severity": issue.get("severity", "unknown"),
                            "recommendation": issue.get("recommendation", "")
                        })
                except json.JSONDecodeError:
                    logger.warning("Could not parse safety check output")
            
        except FileNotFoundError:
            logger.warning("safety command not found. Install with: pip install safety")
        except Exception as e:
            logger.error(f"Python security check failed: {e}")
            self.update_report["errors"].append(f"Python security check: {e}")
        
        return security_issues
    
    def check_node_security(self, package_json_path: Path) -> List[Dict[str, Any]]:
        """Check Node.js dependencies for security vulnerabilities."""
        security_issues = []
        
        try:
            if not package_json_path.exists():
                return security_issues
            
            # Run npm audit
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                cwd=package_json_path.parent
            )
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    
                    if "vulnerabilities" in audit_data:
                        for package, vuln_info in audit_data["vulnerabilities"].items():
                            security_issues.append({
                                "type": "node",
                                "package": package,
                                "severity": vuln_info.get("severity", "unknown"),
                                "title": vuln_info.get("title", ""),
                                "recommendation": vuln_info.get("recommendation", "")
                            })
                            
                except json.JSONDecodeError:
                    logger.warning("Could not parse npm audit output")
                    
        except FileNotFoundError:
            logger.warning("npm command not found")
        except Exception as e:
            logger.error(f"Node.js security check failed: {e}")
            self.update_report["errors"].append(f"Node.js security check: {e}")
        
        return security_issues
    
    def get_outdated_python_packages(self) -> List[Dict[str, Any]]:
        """Get list of outdated Python packages."""
        outdated_packages = []
        
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout:
                packages = json.loads(result.stdout)
                for package in packages:
                    outdated_packages.append({
                        "name": package["name"],
                        "current_version": package["version"],
                        "latest_version": package["latest_version"],
                        "type": package.get("latest_filetype", "wheel")
                    })
                    
        except Exception as e:
            logger.error(f"Failed to check outdated Python packages: {e}")
            self.update_report["errors"].append(f"Python outdated check: {e}")
        
        return outdated_packages
    
    def get_outdated_node_packages(self, package_json_path: Path) -> List[Dict[str, Any]]:
        """Get list of outdated Node.js packages."""
        outdated_packages = []
        
        try:
            if not package_json_path.exists():
                return outdated_packages
            
            result = subprocess.run(
                ["npm", "outdated", "--json"],
                capture_output=True,
                text=True,
                cwd=package_json_path.parent
            )
            
            if result.stdout:
                try:
                    packages = json.loads(result.stdout)
                    for package_name, package_info in packages.items():
                        outdated_packages.append({
                            "name": package_name,
                            "current_version": package_info.get("current", "unknown"),
                            "wanted_version": package_info.get("wanted", "unknown"),
                            "latest_version": package_info.get("latest", "unknown"),
                            "location": package_info.get("location", "")
                        })
                except json.JSONDecodeError:
                    logger.warning("Could not parse npm outdated output")
                    
        except Exception as e:
            logger.error(f"Failed to check outdated Node.js packages: {e}")
            self.update_report["errors"].append(f"Node.js outdated check: {e}")
        
        return outdated_packages
    
    def update_python_packages(self, packages: List[str] = None, 
                             upgrade_all: bool = False) -> List[Dict[str, Any]]:
        """Update specified Python packages or all outdated packages."""
        updates = []
        
        try:
            if upgrade_all:
                # Get all outdated packages
                outdated = self.get_outdated_python_packages()
                packages = [pkg["name"] for pkg in outdated]
            
            if not packages:
                logger.info("No Python packages to update")
                return updates
            
            for package in packages:
                try:
                    # Update package
                    result = subprocess.run(
                        ["pip", "install", "--upgrade", package],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root
                    )
                    
                    if result.returncode == 0:
                        updates.append({
                            "package": package,
                            "status": "success",
                            "output": result.stdout.strip()
                        })
                        logger.info(f"Updated Python package: {package}")
                    else:
                        updates.append({
                            "package": package,
                            "status": "failed",
                            "error": result.stderr.strip()
                        })
                        logger.error(f"Failed to update {package}: {result.stderr}")
                        
                except Exception as e:
                    updates.append({
                        "package": package,
                        "status": "error",
                        "error": str(e)
                    })
                    logger.error(f"Error updating {package}: {e}")
            
        except Exception as e:
            logger.error(f"Python package update failed: {e}")
            self.update_report["errors"].append(f"Python update: {e}")
        
        return updates
    
    def update_node_packages(self, package_json_path: Path, 
                           packages: List[str] = None,
                           upgrade_all: bool = False) -> List[Dict[str, Any]]:
        """Update specified Node.js packages or all outdated packages."""
        updates = []
        
        try:
            if not package_json_path.exists():
                return updates
            
            if upgrade_all:
                # Update all packages
                result = subprocess.run(
                    ["npm", "update"],
                    capture_output=True,
                    text=True,
                    cwd=package_json_path.parent
                )
                
                if result.returncode == 0:
                    updates.append({
                        "package": "all",
                        "status": "success",
                        "output": result.stdout.strip()
                    })
                    logger.info("Updated all Node.js packages")
                else:
                    updates.append({
                        "package": "all",
                        "status": "failed",
                        "error": result.stderr.strip()
                    })
                    logger.error(f"Failed to update Node.js packages: {result.stderr}")
            
            elif packages:
                for package in packages:
                    try:
                        result = subprocess.run(
                            ["npm", "install", f"{package}@latest"],
                            capture_output=True,
                            text=True,
                            cwd=package_json_path.parent
                        )
                        
                        if result.returncode == 0:
                            updates.append({
                                "package": package,
                                "status": "success",
                                "output": result.stdout.strip()
                            })
                            logger.info(f"Updated Node.js package: {package}")
                        else:
                            updates.append({
                                "package": package,
                                "status": "failed",
                                "error": result.stderr.strip()
                            })
                            logger.error(f"Failed to update {package}: {result.stderr}")
                            
                    except Exception as e:
                        updates.append({
                            "package": package,
                            "status": "error",
                            "error": str(e)
                        })
                        logger.error(f"Error updating {package}: {e}")
            
        except Exception as e:
            logger.error(f"Node.js package update failed: {e}")
            self.update_report["errors"].append(f"Node.js update: {e}")
        
        return updates
    
    def generate_requirements_freeze(self) -> None:
        """Generate updated requirements.txt file."""
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Backup current requirements
                if self.requirements_file.exists():
                    backup_path = Path(f"{self.requirements_file}.backup")
                    self.requirements_file.rename(backup_path)
                
                # Write new requirements
                with open(self.requirements_file, 'w') as f:
                    f.write(result.stdout)
                
                logger.info("Updated requirements.txt")
                
        except Exception as e:
            logger.error(f"Failed to update requirements.txt: {e}")
            self.update_report["errors"].append(f"Requirements freeze: {e}")
    
    def full_dependency_check(self) -> Dict[str, Any]:
        """Perform complete dependency check and security audit."""
        logger.info("Starting full dependency check")
        
        # Check security issues
        python_security = self.check_python_security()
        node_security = self.check_node_security(self.package_json)
        frontend_security = self.check_node_security(self.frontend_package_json)
        
        self.update_report["security_issues"].extend(python_security)
        self.update_report["security_issues"].extend(node_security)
        self.update_report["security_issues"].extend(frontend_security)
        
        # Get outdated packages
        outdated_python = self.get_outdated_python_packages()
        outdated_node = self.get_outdated_node_packages(self.package_json)
        outdated_frontend = self.get_outdated_node_packages(self.frontend_package_json)
        
        self.update_report["outdated_packages"] = {
            "python": outdated_python,
            "node_root": outdated_node,
            "node_frontend": outdated_frontend
        }
        
        logger.info(f"Dependency check completed: "
                   f"{len(python_security + node_security + frontend_security)} security issues, "
                   f"{len(outdated_python + outdated_node + outdated_frontend)} outdated packages")
        
        return self.update_report


def main():
    """Main dependency update script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiskX Dependency Update Utility")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for updates, don't install")
    parser.add_argument("--update-python", action="store_true",
                       help="Update Python packages")
    parser.add_argument("--update-node", action="store_true",
                       help="Update Node.js packages")
    parser.add_argument("--packages", nargs="+",
                       help="Specific packages to update")
    parser.add_argument("--report-file", type=str,
                       help="Save report to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    updater = DependencyUpdater()
    
    if args.check_only:
        # Just check for updates and security issues
        report = updater.full_dependency_check()
        
        print(f"Dependency Check Results:")
        print(f"  Security issues: {len(report['security_issues'])}")
        print(f"  Outdated Python packages: {len(report['outdated_packages']['python'])}")
        print(f"  Outdated Node.js packages (root): {len(report['outdated_packages']['node_root'])}")
        print(f"  Outdated Node.js packages (frontend): {len(report['outdated_packages']['node_frontend'])}")
        
        if report["security_issues"]:
            print("\nSecurity Issues Found:")
            for issue in report["security_issues"]:
                print(f"  - {issue['package']} ({issue['type']}): {issue.get('vulnerability', issue.get('title', 'Unknown'))}")
        
    else:
        # Perform updates
        if args.update_python:
            python_updates = updater.update_python_packages(
                packages=args.packages,
                upgrade_all=(not args.packages)
            )
            updater.update_report["python_updates"] = python_updates
            
            # Update requirements.txt
            updater.generate_requirements_freeze()
        
        if args.update_node:
            node_updates = updater.update_node_packages(
                updater.package_json,
                packages=args.packages,
                upgrade_all=(not args.packages)
            )
            frontend_updates = updater.update_node_packages(
                updater.frontend_package_json,
                packages=args.packages,
                upgrade_all=(not args.packages)
            )
            
            updater.update_report["node_updates"] = {
                "root": node_updates,
                "frontend": frontend_updates
            }
        
        print(f"Update completed:")
        if args.update_python:
            print(f"  Python packages updated: {len(updater.update_report['python_updates'])}")
        if args.update_node:
            root_count = len(updater.update_report["node_updates"]["root"])
            frontend_count = len(updater.update_report["node_updates"]["frontend"])
            print(f"  Node.js packages updated: {root_count + frontend_count}")
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            json.dump(updater.update_report, f, indent=2)
        print(f"Report saved to: {args.report_file}")


if __name__ == "__main__":
    main()