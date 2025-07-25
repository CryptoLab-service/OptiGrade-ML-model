# Security Policy

## Supported Versions

The following versions of OptiGrade receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | ✅ Active support |
| 1.x     | ⚠️ Security fixes only |
| < 1.0   | ❌ Not supported |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability, please follow responsible disclosure:

1. **Email Us**: Contact **oluwalowojohn@gmail.com** with:
   - Detailed vulnerability description
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes

2. **Expectations**:
   - We acknowledge reports within **48 hours**
   - Provide regular status updates
   - Work together on mitigation strategy
   - Public disclosure coordinated after fix is released

⚠️ **Do not disclose vulnerabilities publicly** until we've had reasonable time to address them.

## Security Best Practices

### Code Security
- All contributions undergo **static code analysis** (Bandit/Semgrep)
- Secure coding practices enforced (OWASP Top 10 compliance)
- Dependencies monitored via **Dependabot**
- Secrets scanning with **GitGuardian**

### Data Protection
- **End-to-end encryption** for sensitive user data
- **Zero-trust architecture** implementation
- **RBAC controls** for data access
- Never store credentials in code/repos

### API Security
- **JWT authentication** with short-lived tokens
- **Strict input validation** on all endpoints
- **Rate limiting** on public APIs
- Regular penetration testing

## Vulnerability Management

- Critical vulnerabilities: Patched within **72 hours**
- High-risk vulnerabilities: Addressed within **1 week**
- Medium/low vulnerabilities: Fixed in next scheduled release

## Rewards

We appreciate ethical security research through:
- Public acknowledgement (with permission)
- Exclusive contributor badges
- Special mention in release notes