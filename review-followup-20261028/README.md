# Review-size follow-up

The source PR now adds 82 lines across four files: 27 configuration lines, 9 parsing lines, 45 regression-test lines and one CI dependency line. Commit `8dbd02e513817445a5f1ac08ad03bfb0a80aba29` only places each of the 24 configuration entries on one line. Every parameter, value type and key order remains identical to the actual H20-tested `eb34e115` configuration.

[The equivalence receipt](config-format-equivalence.json) records both raw-file hashes and the common canonical JSON hash. Preserving the parsed mapping and insertion order preserves nearest-key selection for every token count, including ties. The generator normally writes expanded JSON; this follow-up intentionally uses compact rows for review and does not change that generator.

[Applicable pre-commit checks](config-format-precommit.stdout.txt) and Git whitespace validation passed. The unchanged parser/test/CI/native sources and equivalent configuration reuse the previously recorded CPU and complete paired GPU results; no model, scorer or old test suite was rerun.

Upstream check diagnosis and original logs are retained in [upstream-checks](upstream-checks/). These records distinguish an authorization failure from a completed code or documentation check.
