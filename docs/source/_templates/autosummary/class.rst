{{ fullname | escape | underline }}

Description
--------------

.. currentmodule:: {{ fullname }}

.. autoclass:: {{ fullname }}
  :members:
  :member-order: bysource

.. .. autoclass:: {{ fullname }}

..   {% if methods %}
..   .. rubric:: Methods

..   .. autosummary::
..     :toctree:

..     {% for method in methods %}
..       {{ method }}
..     {% endfor %}

..   {% endif %}

..   {% if attributes %}
..   .. rubric:: Attributes

..   .. autosummary::
..     :toctree:

..     {% for attribute in attributes %}
..       {{ attribute }}
..     {% endfor %}

..   {% endif %}

