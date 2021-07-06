{{ fullname | escape | underline }}

Description
--------------

.. currentmodule:: {{ fullname | escape }}

.. automodule:: {{ fullname | escape }}

  {% if attributes %}
  Attributes
  -----------

  .. autosummary::
    :toctree:

    {% for attribute in attributes %}
      {{ attribute }}
    {% endfor %}

  {% endif %}

  {% if modules %}
  Modules
  -------

  .. autosummary::
    :toctree:

    {% for module in modules %}
      {{ module }}
    {% endfor %}

  {% endif %}

  {% if classes %}
  Classes
  -------
  .. autosummary::
    :toctree:
    :nosignatures:

      {% for class in classes %}
          {{ class }}
      {% endfor %}

  {% endif %}

  {% if functions %}
  Functions
  ---------
  .. autosummary::
    :toctree:

    {% for function in functions %}
        {{ function }}
    {% endfor %}

  {% endif %}
