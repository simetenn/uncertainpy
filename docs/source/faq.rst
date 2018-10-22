.. _faq:

Frequently asked questions
==========================

Here is a collection of frequently asked questions.


Is Uncertainpy usable with multiple model outputs?
--------------------------------------------------

Yes, however it does unfortunately not have direct support for this.
Uncertainpy by default only performs an uncertainty quantification of the first
model output returned.
But you can return the additional model outputs in the
info dictionary,
and then define new features that extract each model output from the info
dictionary, see the code example in :ref:`Multiple model outputs <multiple_outputs>`.


