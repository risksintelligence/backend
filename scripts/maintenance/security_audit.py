#!/usr/bin/env python3
"""
Security audit utilities for RiskX platform.
Performs comprehensive security checks on code, dependencies, and configuration.
"""

import os
import sys
import json
import subprocess
import datetime
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class SecurityAuditor:
    """Security audit and vulnerability scanning for RiskX platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.audit_report = {
            "audit_date": datetime.datetime.now().isoformat(),
            "code_issues": [],
            "dependency_vulnerabilities": [],
            "configuration_issues": [],
            "secret_scan_results": [],
            "file_permission_issues": [],
            "summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            },
            "errors": []
        }
        
        # Patterns for detecting secrets
        self.secret_patterns = [
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', "API Key"),
            (r'secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', "Secret Key"),
            (r'password["\']?\s*[:=]\s*["\']?([^\s"\']{8,})', "Password"),
            (r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', "Token"),
            (r'aws[_-]?access[_-]?key["\']?\s*[:=]\s*["\']?([A-Z0-9]{20})', "AWS Access Key"),
            (r'aws[_-]?secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})', "AWS Secret Key"),
            (r'database[_-]?url["\']?\s*[:=]\s*["\']?(postgresql://[^\s"\']+)', "Database URL"),
            (r'redis[_-]?url["\']?\s*[:=]\s*["\']?(redis://[^\s"\']+)', "Redis URL"),
        ]
    
    def scan_code_with_bandit(self) -> List[Dict[str, Any]]:
        """Scan Python code for security issues using bandit."""
        issues = []
        
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    for result_item in bandit_data.get("results", []):
                        issues.append({
                            "tool": "bandit",
                            "file": result_item.get("filename", "unknown"),
                            "line": result_item.get("line_number", 0),
                            "severity": result_item.get("issue_severity", "unknown"),
                            "confidence": result_item.get("issue_confidence", "unknown"),
                            "issue": result_item.get("issue_text", ""),
                            "test_id": result_item.get("test_id", ""),
                            "test_name": result_item.get("test_name", "")
                        })
                        
                except json.JSONDecodeError:
                    logger.warning("Could not parse bandit output")
                    
        except FileNotFoundError:
            logger.warning("bandit not found. Install with: pip install bandit")
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            self.audit_report["errors"].append(f"Bandit scan: {e}")
        
        return issues
    
    def scan_dependencies(self) -> List[Dict[str, Any]]:
        """Scan dependencies for known vulnerabilities."""
        vulnerabilities = []
        
        # Python dependencies with safety
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerabilities.append({
                            "type": "python",
                            "package": vuln.get("package", "unknown"),
                            "vulnerability_id": vuln.get("vulnerability_id", ""),
                            "severity": vuln.get("severity", "unknown"),
                            "vulnerable_spec": vuln.get("vulnerable_spec", ""),
                            "description": vuln.get("vulnerability", "")
                        })
                except json.JSONDecodeError:
                    logger.warning("Could not parse safety output")
                    
        except FileNotFoundError:
            logger.warning("safety not found. Install with: pip install safety")
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            self.audit_report["errors"].append(f"Safety scan: {e}")
        
        # Node.js dependencies with npm audit
        for package_json in [self.project_root / "package.json", 
                            self.project_root / "frontend" / "package.json"]:
            if package_json.exists():
                try:
                    result = subprocess.run(
                        ["npm", "audit", "--json"],
                        capture_output=True,
                        text=True,
                        cwd=package_json.parent
                    )
                    
                    if result.stdout:
                        try:
                            audit_data = json.loads(result.stdout)
                            
                            if "vulnerabilities" in audit_data:
                                for package, vuln_info in audit_data["vulnerabilities"].items():
                                    vulnerabilities.append({
                                        "type": "node",
                                        "package": package,
                                        "severity": vuln_info.get("severity", "unknown"),
                                        "title": vuln_info.get("title", ""),
                                        "via": vuln_info.get("via", []),
                                        "effects": vuln_info.get("effects", []),
                                        "location": str(package_json.parent)
                                    })
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse npm audit output for {package_json}")
                            
                except FileNotFoundError:
                    logger.warning("npm not found")
                except Exception as e:
                    logger.error(f"npm audit failed for {package_json}: {e}")
                    self.audit_report["errors"].append(f"npm audit ({package_json}): {e}")
        
        return vulnerabilities
    
    def scan_for_secrets(self) -> List[Dict[str, Any]]:
        """Scan for potential secrets in code and configuration files."""
        secrets_found = []
        
        # Files to scan
        scan_extensions = ['.py', '.js', '.ts', '.tsx', '.json', '.yml', '.yaml', '.env', '.conf']
        exclude_dirs = {'node_modules', 'venv', '__pycache__', '.git', 'data', 'logs'}
        
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in scan_extensions and
                not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs)):
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        for pattern, secret_type in self.secret_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                secrets_found.append({
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "line": line_num,
                                    "type": secret_type,
                                    "pattern": pattern,
                                    "severity": "high" if secret_type in ["API Key", "Secret Key", "AWS Secret Key"] else "medium"
                                })
                                
                except Exception as e:
                    logger.debug(f"Could not scan file {file_path}: {e}")
        
        return secrets_found
    
    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check for insecure file permissions."""
        permission_issues = []
        
        # Critical files that should have restricted permissions
        critical_files = [
            ".env",
            "config/*.conf",
            "scripts/**/*.py",
            "src/**/*.py"
        ]
        
        for pattern in critical_files:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        stat_info = file_path.stat()
                        perms = oct(stat_info.st_mode)[-3:]
                        
                        # Check for world-writable files
                        if int(perms[2]) & 2:  # World writable
                            permission_issues.append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "permissions": perms,
                                "issue": "World writable",
                                "severity": "high"
                            })
                        
                        # Check for overly permissive files
                        if file_path.suffix == '.py' and perms != '644' and perms != '755':
                            if int(perms) > 755:
                                permission_issues.append({
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "permissions": perms,
                                    "issue": "Overly permissive",
                                    "severity": "medium"
                                })
                                
                    except Exception as e:
                        logger.debug(f"Could not check permissions for {file_path}: {e}")
        
        return permission_issues
    
    def check_configuration_security(self) -> List[Dict[str, Any]]:
        """Check configuration files for security issues."""
        config_issues = []
        
        # Check docker-compose.yml
        docker_compose = self.project_root / "docker-compose.yml"
        if docker_compose.exists():
            try:
                with open(docker_compose, 'r') as f:
                    content = f.read()
                    
                    # Check for hardcoded passwords
                    if re.search(r'password:\s*["\']?[^"\'\s]+["\']?', content, re.IGNORECASE):
                        config_issues.append({
                            "file": "docker-compose.yml",
                            "issue": "Potential hardcoded password",
                            "severity": "high",
                            "description": "Docker compose file may contain hardcoded passwords"
                        })
                    
                    # Check for privileged containers
                    if "privileged: true" in content:
                        config_issues.append({
                            "file": "docker-compose.yml",
                            "issue": "Privileged container",
                            "severity": "medium",
                            "description": "Container running in privileged mode"
                        })
                        
            except Exception as e:
                logger.error(f"Could not check docker-compose.yml: {e}")
        
        # Check for .env files with sensitive data
        for env_file in self.project_root.glob("**/.env*"):
            if env_file.is_file():
                try:
                    with open(env_file, 'r') as f:
                        content = f.read()
                        
                        # Check if .env is in version control (should be in .gitignore)
                        gitignore = self.project_root / ".gitignore"
                        if gitignore.exists():
                            with open(gitignore, 'r') as gf:
                                gitignore_content = gf.read()
                                if ".env" not in gitignore_content:
                                    config_issues.append({
                                        "file": str(env_file.relative_to(self.project_root)),
                                        "issue": "Environment file not in .gitignore",
                                        "severity": "high",
                                        "description": "Environment files should be excluded from version control"
                                    })
                        
                except Exception as e:
                    logger.error(f"Could not check {env_file}: {e}")
        
        return config_issues
    
    def categorize_severity(self, issues: List[Dict[str, Any]]) -> None:
        """Categorize issues by severity and update summary."""
        for issue in issues:
            severity = issue.get("severity", "medium").lower()
            
            if severity == "critical":
                self.audit_report["summary"]["critical_issues"] += 1
            elif severity == "high":
                self.audit_report["summary"]["high_issues"] += 1
            elif severity == "medium":
                self.audit_report["summary"]["medium_issues"] += 1
            elif severity == "low":
                self.audit_report["summary"]["low_issues"] += 1
            
            self.audit_report["summary"]["total_issues"] += 1
    
    def generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if self.audit_report["secret_scan_results"]:
            recommendations.append("Remove hardcoded secrets and use environment variables")
            recommendations.append("Implement proper secret management (e.g., HashiCorp Vault)")
        
        if self.audit_report["dependency_vulnerabilities"]:
            recommendations.append("Update vulnerable dependencies to secure versions")
            recommendations.append("Implement automated dependency scanning in CI/CD")
        
        if self.audit_report["file_permission_issues"]:
            recommendations.append("Fix file permissions to follow least privilege principle")
        
        if self.audit_report["configuration_issues"]:
            recommendations.append("Review and secure configuration files")
            recommendations.append("Ensure sensitive configuration is not in version control")
        
        # General recommendations
        recommendations.extend([
            "Implement regular security audits and vulnerability scanning",
            "Use HTTPS for all external communications",
            "Implement proper input validation and sanitization",
            "Enable security headers in web applications",
            "Implement proper logging and monitoring for security events"
        ])
        
        return recommendations
    
    def full_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        logger.info("Starting comprehensive security audit")
        
        # Perform all security checks
        self.audit_report["code_issues"] = self.scan_code_with_bandit()
        self.audit_report["dependency_vulnerabilities"] = self.scan_dependencies()
        self.audit_report["secret_scan_results"] = self.scan_for_secrets()
        self.audit_report["file_permission_issues"] = self.check_file_permissions()
        self.audit_report["configuration_issues"] = self.check_configuration_security()
        
        # Categorize all issues
        all_issues = (
            self.audit_report["code_issues"] +
            self.audit_report["dependency_vulnerabilities"] +
            self.audit_report["secret_scan_results"] +
            self.audit_report["file_permission_issues"] +
            self.audit_report["configuration_issues"]
        )
        
        self.categorize_severity(all_issues)
        
        # Generate recommendations
        self.audit_report["recommendations"] = self.generate_recommendations()
        
        logger.info(f"Security audit completed: {self.audit_report['summary']['total_issues']} issues found")
        
        return self.audit_report


def main():
    """Main security audit script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiskX Security Audit Utility")
    parser.add_argument("--report-file", type=str,
                       help="Save audit report to JSON file")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    auditor = SecurityAuditor()
    
    # Perform security audit
    report = auditor.full_security_audit()
    
    # Output results
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("Security Audit Report")
        print("=" * 50)
        print(f"Audit Date: {report['audit_date']}")
        print(f"Total Issues: {report['summary']['total_issues']}")
        print(f"  Critical: {report['summary']['critical_issues']}")
        print(f"  High: {report['summary']['high_issues']}")
        print(f"  Medium: {report['summary']['medium_issues']}")
        print(f"  Low: {report['summary']['low_issues']}")
        
        if report["code_issues"]:
            print(f"\nCode Issues ({len(report['code_issues'])}):")
            for issue in report["code_issues"][:5]:  # Show first 5
                print(f"  - {issue['file']}:{issue['line']} - {issue['issue']} [{issue['severity']}]")
        
        if report["dependency_vulnerabilities"]:
            print(f"\nDependency Vulnerabilities ({len(report['dependency_vulnerabilities'])}):")
            for vuln in report["dependency_vulnerabilities"][:5]:  # Show first 5
                print(f"  - {vuln['package']} ({vuln['type']}) - {vuln.get('title', vuln.get('description', 'Unknown'))} [{vuln['severity']}]")
        
        if report["secret_scan_results"]:
            print(f"\nPotential Secrets ({len(report['secret_scan_results'])}):")
            for secret in report["secret_scan_results"][:5]:  # Show first 5
                print(f"  - {secret['file']}:{secret['line']} - {secret['type']} [{secret['severity']}]")
        
        if report["recommendations"]:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report["recommendations"][:10], 1):  # Show first 10
                print(f"  {i}. {rec}")
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report_file}")
    
    # Exit with error code if critical or high severity issues found
    if report["summary"]["critical_issues"] > 0 or report["summary"]["high_issues"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()