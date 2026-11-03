# Rust workspace notice follow-up

This supplements `RUST-DEPENDENCIES.md` without changing its resolved package graph. Repository URLs come from cached `Cargo.toml`; commits and package paths come from packaged `.cargo_vcs_info.json`. Only bounded public HTTPS GETs of upstream license/notice files were used. No upstream code was executed. This is evidence collection, not license certification or a determination of distribution obligations.

Exact duplicate files are referenced once. Cached sibling-package texts are reused only with the same repository and packaged commit, and a byte-identical pinned workspace-root file. SHA-256 values below identify the original bytes, including whitespace. A verified workspace reference does not mean that all crate-specific notices have been found.

## Package provenance

| Package | Repository | Packaged commit | `path_in_vcs` | Verified workspace reference |
| --- | --- | --- | --- | --- |
| `block2 0.6.2` | `https://github.com/madsmtm/objc2` | `b4167b582b2f75f9a1be75495c41b765344fd03c` | `crates/block2` | Objective-C workspace statement below; terms/copyright gap remains |
| `candle-kernels 0.11.0` | `https://github.com/huggingface/candle` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | `candle-kernels` | Candle and ug root licenses below |
| `candle-metal-kernels 0.11.0` | `https://github.com/huggingface/candle` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | `candle-metal-kernels` | Candle and ug root licenses below |
| `candle-nn 0.11.0` | `https://github.com/huggingface/candle` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | `candle-nn` | Candle and ug root licenses below |
| `candle-ug 0.11.0` | `https://github.com/huggingface/candle` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | `candle-ug` | Candle and ug root licenses below |
| `dispatch2 0.3.1` | `https://github.com/madsmtm/objc2` | `8852b424193ca41602281b3d7540d7c8ed51e49a` | `crates/dispatch2` | Objective-C workspace statement below; terms/copyright gap remains |
| `objc2 0.6.4` | `https://github.com/madsmtm/objc2` | `8852b424193ca41602281b3d7540d7c8ed51e49a` | `crates/objc2` | Objective-C workspace statement below; terms/copyright gap remains |
| `objc2-core-foundation 0.3.2` | `https://github.com/madsmtm/objc2` | `7b1abfd750a2cacaea71d6a56ecfb83cb7de560b` | `framework-crates/objc2-core-foundation` | Objective-C workspace statement below; terms/copyright gap remains |
| `objc2-encode 4.1.0` | `https://github.com/madsmtm/objc2` | `8d214f5477365ffcbcbb7de058c86ed9a518efb7` | `crates/objc2-encode` | Objective-C workspace statement below; terms/copyright gap remains |
| `objc2-foundation 0.3.2` | `https://github.com/madsmtm/objc2` | `7b1abfd750a2cacaea71d6a56ecfb83cb7de560b` | `framework-crates/objc2-foundation` | Objective-C workspace statement below; terms/copyright gap remains |
| `objc2-metal 0.3.2` | `https://github.com/madsmtm/objc2` | `7b1abfd750a2cacaea71d6a56ecfb83cb7de560b` | `framework-crates/objc2-metal` | Objective-C workspace statement below; terms/copyright gap remains |
| `pulp-wasm-simd-flag 0.1.1` | `https://github.com/sarah-quinones/pulp/` | `5eb07fd7b68edf0a5e19f71737d315f72a510295` | `pulp-wasm-simd-flag` | Pulp root license below |
| `r-efi 5.3.0` | `https://github.com/r-efi/r-efi` | `97b55bed1c2c91dcbf787674849f05337ff80b33` | empty (repository root) | r-efi AUTHORS below |
| `ug 0.5.0` | `https://github.com/LaurentMazare/ug` | `8c6dd50d6e96a22db70e1462c0e49d0cda8294f7` | `ug-core` | Candle and ug root licenses below |
| `ug-cuda 0.5.0` | `https://github.com/LaurentMazare/ug` | `8c6dd50d6e96a22db70e1462c0e49d0cda8294f7` | `ug-cuda` | Candle and ug root licenses below |
| `ug-metal 0.5.0` | `https://github.com/LaurentMazare/ug` | `8c6dd50d6e96a22db70e1462c0e49d0cda8294f7` | `ug-metal` | Candle and ug root licenses below |
| `windows-link 0.2.1` | `https://github.com/microsoft/windows-rs` | `d468916ac27a36fb8a12bafc1bf5c0ec2fe92238` | `crates/libs/link` | Windows root licenses below |
| `windows-sys 0.61.2` | `https://github.com/microsoft/windows-rs` | `32c3144490c016fe496a0aed769bce60987a2e9d` | `crates/libs/sys` | Windows root licenses below |

## Remaining gaps and scope

Of the original 18 cache-pattern gaps, nine now have upstream license text, seven have only the verified Objective-C workspace statement, and two remain unpinned.

- Native macOS runtime: `block 0.1.6` and `malloc_buf 0.0.6` have no packaged `.cargo_vcs_info.json`. Their cached repositories are respectively `http://github.com/SSheldon/rust-block` and `https://github.com/SSheldon/malloc_buf`; neither declaration establishes a source commit or `path_in_vcs`. No current-branch or guessed-tag license is substituted.
- Native macOS runtime: `block2`, `dispatch2`, `objc2`, `objc2-core-foundation`, `objc2-encode`, `objc2-foundation`, and `objc2-metal` have verified root `LICENSE.md` statements, but those statements link to generic license terms and contain no crate-specific copyright notice. Full terms/copyright provenance remains unresolved by this bounded root-file follow-up. The upstream Apple SDK caveat is retained verbatim, not assessed or certified.
- Candle NN/Metal/ug and ug/ug-metal: pinned workspace MIT and Apache texts are collected. This closes the missing workspace-text entries, not a comprehensive audit of embedded kernels, SDKs, or other source-file attributions.
- Graph extras: `pulp-wasm-simd-flag` has the verified Pulp workspace MIT text; its name alone is not evidence that it is absent from native builds (Pulp declares it as a dependency). `r-efi` is gated by `cfg(all(target_os = "uefi", getrandom_backend = "efi_rng"))` in cached `getrandom 0.3.4`, not the `darwin-arm64` release bundle. Its Cargo declaration offers MIT OR Apache-2.0 OR LGPL-2.1-or-later. LGPL is an alternative, not required copyleft. The collected AUTHORS file includes the MIT permission text and preserves the other alternatives without choosing one. Windows workspace licenses are collected for target-specific graph entries outside that macOS bundle. No remaining gap is classified as merely a build-only dependency without evidence.

The root `NOTICE` probes for Candle, ug, all four Objective-C commits, and both Windows commits returned HTTP 404. Additional Candle/ug `NOTICE.txt`, `NOTICE.md`, and `COPYRIGHT` probes returned 404; Pulp `NOTICE` and `COPYRIGHT` probes returned 404. These limited probes do not establish that notices are absent elsewhere. No notice was invented. Successful source URLs are recorded below; failed license filename guesses are not treated as evidence of license coverage.

## Candle and ug root licenses

Sources (the commit is also embedded in each URL):

- `https://raw.githubusercontent.com/huggingface/candle/31f35b147389700ed2a178ee66a91c3cc25cc80d/LICENSE-APACHE`
- `https://raw.githubusercontent.com/huggingface/candle/31f35b147389700ed2a178ee66a91c3cc25cc80d/LICENSE-MIT`
- `https://raw.githubusercontent.com/LaurentMazare/ug/8c6dd50d6e96a22db70e1462c0e49d0cda8294f7/LICENSE-APACHE`
- `https://raw.githubusercontent.com/LaurentMazare/ug/8c6dd50d6e96a22db70e1462c0e49d0cda8294f7/LICENSE-MIT`

Both `LICENSE-APACHE` files have SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`, identical to cached `candle-core 0.11.0/LICENSE` already preserved as **Upstream text 16** in `RUST-DEPENDENCIES.md`. The cached `candle-core` package has repository `https://github.com/huggingface/candle`, commit `31f35b147389700ed2a178ee66a91c3cc25cc80d`, and path `candle-core`, matching the other Candle packages. Reuse does not rely on version numbers alone.

Both `LICENSE-MIT` files have SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`. Their identical text is preserved once here. It is not byte-identical to cached allocator-api2's similarly worded license.

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## Objective-C workspace statement

Sources, each fetched at its own packaged commit (identical files are included once):

- `https://raw.githubusercontent.com/madsmtm/objc2/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md`
- `https://raw.githubusercontent.com/madsmtm/objc2/8852b424193ca41602281b3d7540d7c8ed51e49a/LICENSE.md`
- `https://raw.githubusercontent.com/madsmtm/objc2/8d214f5477365ffcbcbb7de058c86ed9a518efb7/LICENSE.md`
- `https://raw.githubusercontent.com/madsmtm/objc2/b4167b582b2f75f9a1be75495c41b765344fd03c/LICENSE.md`

All four files have SHA-256 `7f976f7e9cb2d87df7230606feb932c3f21ac0e664045a775b600046ff850c54`, matching the verbatim block below. This is the upstream licensing statement, not a replacement for missing license terms or crate-specific copyright notices. Its external references are preserved as text; no mutable external license was substituted for a pinned upstream file.

```text
# License

The licensing of these crates is a bit complicated:
- The crates `objc2`, `block2`, `objc2-foundation` and `objc2-encode` are
  [currently][#23] licensed under [the MIT license][MIT].
- All other crates are trio-licensed under the [Zlib], [Apache-2.0] or [MIT]
  license, at your option.

Furthermore, the crates are (usually automatically) derived from Apple SDKs,
and that may have implications for licensing, see below for details.

[#23]: https://github.com/madsmtm/objc2/issues/23
[MIT]: https://opensource.org/license/MIT
[Zlib]: https://zlib.net/zlib_license.html
[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0


## Apple SDKs

These crates are derived from Apple SDKs shipped with Xcode. You can obtain a
copy of the Xcode license at:

https://www.apple.com/legal/sla/docs/xcode.pdf

Or by typing `xcodebuild -license` in your terminal.

From reading the license, it is unclear whether distributing derived works
such as these crates are allowed?

But in any case, to practically use these crates, you will have to link, and
that only works when you have the correct Xcode SDK available to provide the
required `.tbd` files, which is why we choose to still use the normal SPDX
identifiers in the crates (Xcode is required to use the crates, and when using
Xcode you have already agreed to the Xcode license).
```

## Pulp root license

Source: `https://raw.githubusercontent.com/sarah-quinones/pulp/5eb07fd7b68edf0a5e19f71737d315f72a510295/LICENSE`.

SHA-256 `d64f878c89bd5f1e5ada7e4aad57690b122727cf5229b7b87ea1980c188da1d9`, identical to cached `pulp 0.22.3/LICENSE`, preserved as **Upstream text 47** in `RUST-DEPENDENCIES.md`. The cached sibling has the same repository and commit, path `pulp`. This verifies the workspace-root source for `pulp-wasm-simd-flag 0.1.1`; no inference from a different Pulp release is used.

## r-efi AUTHORS

Source: `https://raw.githubusercontent.com/r-efi/r-efi/97b55bed1c2c91dcbf787674849f05337ff80b33/AUTHORS`.

SHA-256 `ff92bed461f50338dd703a9ba9aee496a425957873df1d91192776e4bdf5dda7`. This is also byte-identical to cached `r-efi 5.3.0/AUTHORS`, which was outside the original license-filename pattern. All alternatives and copyright/authorship text are retained verbatim.

```text
LICENSE:
        This project is triple-licensed under the MIT License, the Apache
        License, Version 2.0, and the GNU Lesser General Public License,
        Version 2.1+.

AUTHORS-MIT:
        Permission is hereby granted, free of charge, to any person obtaining a
        copy of this software and associated documentation files (the
        "Software"), to deal in the Software without restriction, including
        without limitation the rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of the Software, and to
        permit persons to whom the Software is furnished to do so, subject to
        the following conditions:

        The above copyright notice and this permission notice shall be included
        in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

AUTHORS-ASL:
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

AUTHORS-LGPL:
        This program is free software; you can redistribute it and/or modify it
        under the terms of the GNU Lesser General Public License as published
        by the Free Software Foundation; either version 2.1 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
        Lesser General Public License for more details.

        You should have received a copy of the GNU Lesser General Public License
        along with this program; If not, see <http://www.gnu.org/licenses/>.

COPYRIGHT: (ordered alphabetically)
        Copyright (C) 2017-2023 Red Hat, Inc.
        Copyright (C) 2019-2023 Microsoft Corporation
        Copyright (C) 2022-2023 David Rheinsberg

AUTHORS: (ordered alphabetically)
        Alex James <theracermaster@gmail.com>
        Ayush Singh <ayushsingh1325@gmail.com>
        Boris-Chengbiao Zhou <bobo1239@web.de>
        Bret Barkelew <bret@corthon.com>
        Christopher Zurcher <christopher.zurcher@microsoft.com>
        David Rheinsberg <david@readahead.eu>
        Dmitry Mostovenko <trueberserker@gmail.com>
        Hiroki Tokunaga <tokusan441@gmail.com>
        Joe Richey <joerichey@google.com>
        John Schock <joschock@microsoft.com>
        Michael Kubacki <michael.kubacki@microsoft.com>
        Oliver Smith-Denny <osde@microsoft.com>
        Richard Wiedenhöft <richard@wiedenhoeft.xyz>
        Rob Bradford <robert.bradford@intel.com>, <rbradford@rivosinc.com>
        Tom Gundersen <teg@jklm.no>
        Trevor Gross <tmgross@umich.edu>
```

## Windows root licenses

Sources, fetched separately at both packaged commits; identical MIT files and identical Apache files are each preserved once:

- `https://raw.githubusercontent.com/microsoft/windows-rs/32c3144490c016fe496a0aed769bce60987a2e9d/license-apache-2.0`
- `https://raw.githubusercontent.com/microsoft/windows-rs/32c3144490c016fe496a0aed769bce60987a2e9d/license-mit`
- `https://raw.githubusercontent.com/microsoft/windows-rs/d468916ac27a36fb8a12bafc1bf5c0ec2fe92238/license-apache-2.0`
- `https://raw.githubusercontent.com/microsoft/windows-rs/d468916ac27a36fb8a12bafc1bf5c0ec2fe92238/license-mit`

### `license-apache-2.0`

Both source files and the block below have SHA-256 `c16f8dcf1a368b83be78d826ea23de4079fe1b4469a0ab9ee20563f37ff3d44b`.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright (c) Microsoft Corporation.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### `license-mit`

Both source files and the block below have SHA-256 `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383`.

```text
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```
