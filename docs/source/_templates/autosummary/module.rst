{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. module:: {{ fullname }}

{% if modules %}
Modules
-------

.. automodule:: {{ fullname }}

.. autosummary::
  :toctree:

  {% for module in modules %}
    {{ module }}
  {% endfor %}

{% else %}

.. automodule:: {{ fullname }}
  :members:
  :member-order: groupwise

{% endif %}

