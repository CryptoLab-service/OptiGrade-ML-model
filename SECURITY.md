# Security Policy

## Supported Versions

The following versions of OptiGrade App receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| 6.x     | ✅ Supported |
| 5.x     | ✅ Supported |
| 4.x     | ❌ Not Supported |
| < 4.0   | ❌ Not Supported |

Older versions are **not maintained**, and users should upgrade to the latest release for security fixes.

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability, please follow the steps below:

- **Contact Us**: Email **oluwalowojohn@gmail.com** with detailed information.
- **GitHub Issues**: Open a **private issue** (if enabled) for security-related discussions.
- **Response Time**: We aim to acknowledge reports within **48 hours**.
- **Next Steps**: If accepted, we will provide mitigation plans, patches, and release updates.

⚠️ **Do not disclose vulnerabilities publicly** until a fix is available.

## Security Best Practices

### **Code Security**
- Contributions must pass **static code analysis** before merging.
- Follow **secure coding practices** to prevent SQL injection, XSS, and other vulnerabilities.
- Dependencies are **regularly updated** to mitigate security risks.

### **Data Protection**
- **User data encryption** is enforced at all levels.
- **Access control** measures prevent unauthorized data access.
- No sensitive credentials should be **hard-coded in the repository**.

### **API & Authentication**
- We implement **OAuth 2.0** and **JWT tokens** for authentication.
- API endpoints undergo **security reviews** before deployment.

## Responsible Disclosure

We encourage ethical disclosure practices and reward contributors who responsibly report security issues. If you find a critical vulnerability, reach out to us through **proper channels**.

---
