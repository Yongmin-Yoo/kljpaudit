# Preparing an anonymous review release

Prepare a separate clean export; do not use the personal GitHub repository
as an anonymous artifact.

1. Check the current ARR and target-venue anonymity and artifact rules.
2. Export only approved code, aggregate tables, synthetic fixtures, and docs.
3. Exclude `.git`, Git history, private workspace reports, datasets, case IDs,
   annotations, predictions, checkpoints, credentials, and notebook outputs.
4. Remove identifying authorship, personal repository URLs, citation records,
   badge links, and identifying notebook/asset metadata from the review copy.
5. Preserve required third-party notices. Use appropriately anonymous project
   metadata where allowed; do not misrepresent third-party ownership.
6. Scan the actual archive contents and use a venue-approved anonymous hosting
   route. Check links and redirects for identifying account information.

Keep this export separate from the personal development branch. Do not
rewrite or delete the original repository history to prepare it.
