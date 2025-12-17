# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in FluxFlow Core, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues via one of these methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to https://github.com/danny-mio/fluxflow-core/security/advisories
   - Click "Report a vulnerability"
   - Provide detailed information about the vulnerability

2. **Email**
   - Send to: danny-mio@libero.it
   - Subject: "[SECURITY] FluxFlow Core - [Brief Description]"
   - Include detailed information (see below)

### What to Include

When reporting a vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: What can an attacker achieve?
- **Affected versions**: Which versions are vulnerable?
- **Reproduction steps**: Step-by-step instructions to reproduce
- **Proof of concept**: Code/commands demonstrating the issue (if applicable)
- **Suggested fix**: If you have ideas on how to fix it (optional)
- **Your contact information**: For follow-up questions

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial assessment**: Within 5 business days
- **Status updates**: Every 7 days until resolved
- **Credit**: You will be credited in the security advisory (unless you prefer to remain anonymous)

### Security Update Process

1. **Validation**: We verify and assess the vulnerability
2. **Fix development**: We develop and test a fix
3. **Coordinated disclosure**: We coordinate a release timeline with you
4. **Release**: Security patch released with advisory
5. **Public disclosure**: After users have time to update (typically 7-14 days)

## Security Best Practices

When using FluxFlow Core:

### Model Files

- ✅ **Only load model files from trusted sources**
- ✅ **Verify checksums** of downloaded models when available
- ❌ **Never load untrusted .safetensors or .pt files**
- ⚠️  Model files can contain malicious code or backdoors

### Input Validation

- ✅ **Validate all user inputs** (prompts, file paths, parameters)
- ✅ **Sanitize file paths** to prevent directory traversal attacks
- ❌ **Never pass unsanitized user input** to system calls or file operations

### Dependencies

- ✅ **Keep dependencies updated** - run `pip install --upgrade fluxflow` regularly
- ✅ **Use virtual environments** to isolate FluxFlow from system Python
- ✅ **Review dependency security advisories** via GitHub Dependabot

### Network Security

- ✅ **Use HTTPS** when downloading models or data
- ✅ **Validate SSL certificates** - don't disable certificate verification
- ⚠️  Be cautious when accepting model URLs from untrusted sources

### Deployment

- ✅ **Run with least privilege** - don't run as root/admin
- ✅ **Restrict file system access** using appropriate permissions
- ✅ **Use resource limits** (memory, GPU) to prevent DoS
- ✅ **Enable logging** for security monitoring

## Known Security Considerations

### PyTorch Model Loading

FluxFlow uses PyTorch and safetensors for model loading. Be aware:

- **Pickle vulnerability**: PyTorch `.pt` files use pickle, which can execute arbitrary code
- **Safetensors**: We prefer `.safetensors` format which is safer
- **Recommendation**: Only load models from trusted sources (Hugging Face official, verified creators)

### GPU Memory

- Models can consume significant GPU memory
- Out-of-memory conditions can crash applications
- Recommendation: Validate model size vs available GPU memory before loading

### Code Execution

- FluxFlow executes Python code for custom models and configurations
- Untrusted code can compromise your system
- Recommendation: Review all custom code before execution

## Security Hardening Checklist

When deploying FluxFlow in production:

- [ ] Run in isolated container/VM
- [ ] Use non-root user
- [ ] Restrict network access (firewall rules)
- [ ] Enable comprehensive logging
- [ ] Implement rate limiting for API endpoints
- [ ] Validate and sanitize all inputs
- [ ] Use latest stable version
- [ ] Monitor security advisories
- [ ] Regular security audits
- [ ] Backup critical data

## Vulnerability Disclosure Policy

We follow responsible disclosure principles:

- **Coordinated disclosure**: We work with reporters to coordinate public disclosure
- **Reasonable timeframe**: We aim to fix critical issues within 30 days
- **Transparency**: We publish security advisories for all confirmed vulnerabilities
- **Credit**: We credit researchers who report vulnerabilities responsibly

## Security Hall of Fame

We recognize security researchers who help keep FluxFlow secure:

<!-- This section will list researchers who have responsibly disclosed vulnerabilities -->

*No vulnerabilities reported yet.*

## Contact

- **Security issues**: danny-mio@libero.it (use [SECURITY] prefix)
- **General questions**: https://github.com/danny-mio/fluxflow-core/discussions
- **Bug reports**: https://github.com/danny-mio/fluxflow-core/issues

## Additional Resources

- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Python Security](https://docs.python.org/3/library/security.html)
- [PyTorch Community Guidelines](https://pytorch.org/docs/stable/community/contribution_guide.html)

---

**Last Updated**: December 14, 2025
